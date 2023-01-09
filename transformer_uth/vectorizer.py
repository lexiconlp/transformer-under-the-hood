from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def _build_idx_vocabulary(corpus: Sequence[str]) -> Dict[str, int]:
    unique_chars = sorted({c for row in corpus for c in row})
    return {c: i for i, c in enumerate(unique_chars)}


def _text2ids(text: str, vocabulary: Dict[str, int]) -> List[int]:
    return [vocabulary[c] for c in text]


def _idxs_to_one_hot(
    idxs: List[int], max_len: int, num_classes: int
) -> torch.Tensor:
    one_hot = torch.zeros(max_len, num_classes)
    for i, idx in enumerate(idxs):
        one_hot[i][idx] = 1
    return one_hot


def _text_to_one_hot(
    text: str, vocabulary: Dict[str, int], max_len: int
) -> torch.Tensor:
    idxs = _text2ids(text, vocabulary)
    return _idxs_to_one_hot(idxs, max_len, len(vocabulary))


class CharVectorizer(Dataset):
    def __init__(self, path_data: Path):
        self.inputs_text, self.outputs_text = zip(
            *[
                row.split("\t")
                for row in path_data.read_text().split("\n")
                if row
            ]
        )

        self.input_vocab = _build_idx_vocabulary(self.inputs_text)
        self.output_vocab = _build_idx_vocabulary(self.outputs_text)

        self.input_max_len = max(len(row) for row in self.inputs_text)
        self.output_max_len = max(len(row) for row in self.outputs_text)

    def __len__(self):
        return len(self.inputs_text)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        input_text = self.inputs_text[idx]
        output_text = self.outputs_text[idx]

        tensor_input = _text_to_one_hot(
            input_text, self.input_vocab, self.input_max_len
        )
        tensor_output = _text_to_one_hot(
            output_text, self.output_vocab, self.output_max_len
        )

        return tensor_input, tensor_output
