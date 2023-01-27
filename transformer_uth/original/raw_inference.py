import json
from pathlib import Path
from typing import List

import torch

from transformer_uth.consts import PATH_MODELS
from transformer_uth.original.raw_pytorch_transformer import Seq2SeqTransformer

SRC_LANGUAGE = "raw_boxes"
TGT_LANGUAGE = "date"
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


def _get_vocab_transform(path_vocab: Path):
    vocab_transform = {}
    src_vocab = json.loads((path_vocab / "src_vocab.json").read_text())
    tgt_vocab = json.loads((path_vocab / "tgt_vocab.json").read_text())
    inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}
    vocab_transform[SRC_LANGUAGE] = lambda x: [src_vocab[c] for c in x]
    vocab_transform[TGT_LANGUAGE] = lambda x: [inv_tgt_vocab[c] for c in x]

    return vocab_transform, src_vocab, tgt_vocab


def _get_text_transform(token_transform, vocab_transform):
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


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for _i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (
            generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        ).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
        )
        if next_word == EOS_IDX:
            break
    return ys


# token_transform
# vocab_transform
# text_transform
# greedy_decode

# to see inside the model
# https://discuss.pytorch.org/t/obtaining-outputs-and-attention-weights-from-intermediate-transformer-layers/74474/3
# https://github.com/yoshitomo-matsubara/torchdistill/blob/main/demo/extract_intermediate_representations.ipynb


def translate(
    model: torch.nn.Module, src_sentence: str, text_transform, vocab_transform
):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    return (
        " ".join(vocab_transform[TGT_LANGUAGE](list(tgt_tokens.cpu().numpy())))
        .replace("<bos>", "")
        .replace("<eos>", "")
    )


def _load_model(path_model: Path, src_size, tgt_size) -> Seq2SeqTransformer:
    src_vocab_size = src_size
    tgt_vocab_size = tgt_size
    emb_size = 512
    nhead = 8
    ffn_hid_dim = 512
    num_encoder_layers = 3
    num_decoder_layers = 3

    model = Seq2SeqTransformer(
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        ffn_hid_dim,
    )
    model.load_state_dict(torch.load(path_model))
    return model


if __name__ == "__main__":
    token_transform = _get_token_transform()
    vocab_transform, src_vocab, tgt_vocab = _get_vocab_transform(PATH_MODELS)
    text_transform = _get_text_transform(token_transform, vocab_transform)

    path_transformer = PATH_MODELS / "text2number_transformer.pth"
    model = _load_model(path_transformer, len(src_vocab), len(tgt_vocab))

    print(translate(model, "veintisiete", text_transform, vocab_transform))
