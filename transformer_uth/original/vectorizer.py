"""
Transformer adaptation from:
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from transformer_uth.consts import PATH_DATA

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their
# indices to properly insert them in vocab
# SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']
SPECIAL_SYMBOLS = {"?": UNK_IDX, "": PAD_IDX, "\t\t": BOS_IDX, "\t": EOS_IDX}

SRC_LANGUAGE = "raw_boxes"
TGT_LANGUAGE = "date"
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_idx_vocabulary(corpus: Sequence[str]) -> Dict[str, int]:
    unique_chars = sorted({c for row in corpus for c in row})
    char2id = SPECIAL_SYMBOLS.copy()
    char2id.update(
        {c: i + len(SPECIAL_SYMBOLS) for i, c in enumerate(unique_chars)}
    )
    return char2id


def _text2ids(
    text: str, vocabulary: Dict[str, int], max_len: int
) -> List[int]:
    # +2 because <bos> and <eos> symbols
    assert len(text) + 2 <= max_len
    return (
        [BOS_IDX]
        + [vocabulary.get(c, UNK_IDX) for c in text]
        + [EOS_IDX]
        + [PAD_IDX for _ in range(len(text) + 2, max_len)]
    )


def _ids2text(idxs: List[int], id_vocabulary: Dict[int, str]) -> str:
    return "".join(id_vocabulary[idx] for idx in idxs)


def _generate_square_subsequent_mask(size_matrix: int):
    """Prevent look into the future 🙈"""
    mask = (
        torch.triu(torch.ones((size_matrix, size_matrix), device=DEVICE)) == 1
    ).transpose(0, 1)
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def create_mask(
    src_idxs: List[int], tgt_idxs: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.tensor(src_idxs).clone().detach()
    tgt = torch.tensor(tgt_idxs).clone().detach()

    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = _generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(
        torch.bool
    )

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class Seq2SeqDataset(Dataset):
    def __init__(self, path_data: Path, num_samples: int = 200):
        self.inputs_text, self.outputs_text = zip(
            *[
                row.split("\t")
                for row in path_data.read_text().split("\n")[:num_samples]
                if row
            ]
        )

        self.input_vocab = _build_idx_vocabulary(self.inputs_text)
        self.output_vocab = _build_idx_vocabulary(self.outputs_text)

        self.input_id2char = {v: k for k, v in self.input_vocab.items()}
        self.output_id2char = {v: k for k, v in self.output_vocab.items()}

        self.input_max_len = max(len(row) for row in self.inputs_text)
        self.output_max_len = max(len(row) for row in self.outputs_text)

    def __len__(self):
        return len(self.inputs_text)

    def __getitem__(self, idx):
        input_text = self.inputs_text[idx]
        output_text = self.outputs_text[idx]

        return self.vectorize_input(input_text), self.vectorize_output(
            output_text
        )

    def vectorize_input(self, text_input: str) -> torch.Tensor:
        # +2 because <bos> and <eos> symbols
        input_idx = _text2ids(
            text_input, self.input_vocab, self.input_max_len + 2
        )
        return torch.tensor(input_idx)

    def vectorize_output(self, text_output: str) -> torch.Tensor:
        # +2 because <bos> and <eos> symbols
        output_idx = _text2ids(
            text_output, self.output_vocab, self.output_max_len + 2
        )
        return torch.tensor(output_idx)

    def ids_output2text(self, ids_output: torch.Tensor) -> str:
        ids_int = list(map(int, ids_output))
        return _ids2text(ids_int, self.output_id2char)


if __name__ == "__main__":
    task_name = "200_es_numbers_translation"
    path_data = PATH_DATA / f"{task_name}.tsv"
    dataset = Seq2SeqDataset(path_data)

    for sample in dataset:
        inputs_ids = sample[0]
        outputs_ids = sample[1]
        print(_ids2text(inputs_ids, dataset.input_id2char))
        print(_ids2text(outputs_ids, dataset.output_id2char))

        break
