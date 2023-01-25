from pathlib import Path

import torch

from transformer_uth.consts import PATH_DATA, PATH_MODELS
from transformer_uth.original.model import (
    EMB_SIZE,
    FFN_HID_DIM,
    NHEAD,
    NUM_DECODER_LAYERS,
    NUM_ENCODER_LAYERS,
    Seq2SeqTransformer,
)
from transformer_uth.original.vectorizer import (
    _generate_square_subsequent_mask,
    BOS_IDX,
    EOS_IDX,
    Seq2SeqDataset,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_vectorizer(path_train_data: Path) -> Seq2SeqDataset:
    return Seq2SeqDataset(path_train_data)


def _load_model(path_train_data: Path, path_model: Path) -> Seq2SeqTransformer:
    train_dataset = _load_vectorizer(path_train_data)

    model = Seq2SeqTransformer(
        len(train_dataset.input_vocab),
        len(train_dataset.output_vocab),
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        FFN_HID_DIM,
    )
    model.load_state_dict(torch.load(path_model))
    return model


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """Generate output sequence using greedy algorithm."""
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for _ in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (
            _generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
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


def translate(
    model: Seq2SeqTransformer, vectorizer: Seq2SeqDataset, src_sentence: str
):
    src = vectorizer.vectorize_input(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    return vectorizer.ids_output2text(tgt_tokens)


if __name__ == "__main__":
    task = "es-numbers-translation-0-221"
    path_train = PATH_DATA / f"{task}.tsv"
    path_model = PATH_MODELS / f"{task}.pth"

    model = _load_model(path_train, path_model)
    dataset = _load_vectorizer(path_train)

    one_text_sample = "ciento veinte"
    print(translate(model, dataset, one_text_sample))
