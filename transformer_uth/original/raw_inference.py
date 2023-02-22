import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from transformer_uth.original.raw_pytorch_transformer import (
    EMB_SIZE,
    FFN_HID_DIM,
    MODEL_NAME,
    NHEAD,
    NUM_DECODER_LAYERS,
    NUM_ENCODER_LAYERS,
    PATH_TRANSFORMER_MODEL,
    Seq2SeqTransformer,
    SRC_LANGUAGE,
    TGT_LANGUAGE,
)

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat(
        (
            torch.tensor([BOS_IDX]),
            torch.tensor(token_ids),
            torch.tensor([EOS_IDX]),
        )
    )


def _get_token_transform():
    token_transform = {}
    token_transform[SRC_LANGUAGE] = lambda x: [c.lower() for c in x]
    token_transform[TGT_LANGUAGE] = lambda x: list(x)
    return token_transform


def _get_vocab_transform():
    vocab_transform = {}
    src_vocab = json.loads(
        (PATH_TRANSFORMER_MODEL / "scr-vocab.json").read_text()
    )
    tgt_vocab = json.loads(
        (PATH_TRANSFORMER_MODEL / "tgt-vocab.json").read_text()
    )
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    vocab_transform[SRC_LANGUAGE] = lambda x: [
        src_vocab.get(c, UNK_IDX) for c in x
    ]
    vocab_transform[TGT_LANGUAGE] = lambda x: [
        inv_tgt_vocab.get(c, UNK_IDX) for c in x
    ]

    return vocab_transform, src_vocab, tgt_vocab


def _get_text_transform(vocab_transform):
    token_transform = _get_token_transform()
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln],
            # Tokenization
            vocab_transform[ln],
            # Numericalization
            tensor_transform,
        )  # Add BOS/EOS and create tensor
    return text_transform


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(
        0, 1
    )
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def _sa_block(
    model_layer,
    x_: Tensor,
    attn_mask: Optional[Tensor],
    key_padding_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    x_, mh_att = model_layer.self_attn(
        x_,
        x_,
        x_,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    return model_layer.dropout1(x_), mh_att


def _mha_block(
    model_layer,
    x_: Tensor,
    mem: Tensor,
    attn_mask: Optional[Tensor],
    key_padding_mask: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    x_, mh_att = model_layer.multihead_attn(
        x_,
        mem,
        mem,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    return model_layer.dropout2(x_), mh_att


def _get_encoder_attention_layers(
    model: Seq2SeqTransformer, input_tks, input_mask
):
    # input_tks = text_transform[SRC_LANGUAGE](input_text).view(-1, 1)
    # num_tks = input_tks.shape[0]
    # input_mask = (torch.zeros(num_tks, num_tks)).type(torch.bool)
    input_emb = model.src_tok_emb(input_tks)
    input_pos_emb = model.positional_encoding(input_emb)

    x_ = input_pos_emb
    enc_attentions = []
    for enc_layer in model.transformer.encoder.layers:
        enc_x, mh_att = _sa_block(enc_layer, x_, input_mask, None)
        enc_attentions.append(mh_att)
        x_ = enc_layer.norm1(x_ + enc_x)
        x_ = enc_layer.norm2(x_ + enc_layer._ff_block(x_))

    return model.transformer.encoder.norm(x_), enc_attentions


def _get_decoder_attention_layers(
    model: Seq2SeqTransformer, pos_input_emb: Tensor, output_mask, ys: Tensor
):
    # output_mask = generate_square_subsequent_mask(ys.size(0)).
    # type(torch.bool)
    output_emb = model.tgt_tok_emb(ys)
    pos_output_emb = model.positional_encoding(output_emb)

    x_ = pos_output_emb
    dec_att = []
    for dec_layer in model.transformer.decoder.layers:
        dec_x, self_att = _sa_block(dec_layer, x_, output_mask, None)
        x_ = dec_layer.norm1(x_ + dec_x)
        dec_x, mh_att = _mha_block(dec_layer, x_, pos_input_emb, None, None)
        dec_att.append(mh_att)
        x_ = dec_layer.norm2(x_ + dec_x)
        x_ = dec_layer.norm3(x_ + dec_layer._ff_block(x_))

    return model.transformer.decoder.norm(x_), dec_att


def greedy_decode(model, src, src_mask, max_len, start_symbol):

    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory, enc_att = _get_encoder_attention_layers(model, src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    dec_att = []
    for _i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (
            generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        ).to(DEVICE)
        out, dec_att_lay = _get_decoder_attention_layers(
            model, memory, tgt_mask, ys
        )
        dec_att.append(dec_att_lay)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
        )
        if next_word == EOS_IDX:
            break
    return ys, enc_att, dec_att


def translate(
    model: torch.nn.Module, src_sentence: str, text_transform, vocab_transform
):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens, enc_att, dec_att = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    )
    tgt_tokens = tgt_tokens.flatten()
    text_output = (
        " ".join(vocab_transform[TGT_LANGUAGE](list(tgt_tokens.cpu().numpy())))
        .replace("<bos>", "")
        .replace("<eos>", "")
    )
    return text_output, enc_att, dec_att


def _load_model(path_model: Path, src_vocab, tgt_vocab) -> Seq2SeqTransformer:
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        len(src_vocab),
        len(tgt_vocab),
        FFN_HID_DIM,
    )

    model.load_state_dict(torch.load(path_model))
    return model


if __name__ == "__main__":
    vocab_transform, src_vocab, tgt_vocab = _get_vocab_transform()
    text_transform = _get_text_transform(vocab_transform)

    path_transformer = PATH_TRANSFORMER_MODEL / MODEL_NAME
    model = _load_model(path_transformer, src_vocab, tgt_vocab)

    output_text, enc_at, dec_at = translate(
        model, "veintisiete", text_transform, vocab_transform
    )
