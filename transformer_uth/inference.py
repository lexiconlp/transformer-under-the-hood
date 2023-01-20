from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformer_uth.attention_models import AttentionModel
from transformer_uth.consts import PATH_DATA, PATH_MODELS, PATH_TEST
from transformer_uth.vectorizer import _text_to_one_hot, CharVectorizer


def _load_model(path_data: Path, path_model: Path) -> AttentionModel:
    dataset = CharVectorizer(path_data)
    model = AttentionModel(dataset.seq_data)
    model.load_state_dict(torch.load(path_model))
    return model


def _evaluate_model(path_test: Path, model: AttentionModel):
    model.eval()
    dataset = CharVectorizer(path_test)
    data_loader = DataLoader(dataset=dataset, batch_size=len(dataset))
    for batch in data_loader:
        minibatch_x, minibatch_y = batch

        logit_pred, *_ = model(minibatch_x)
        pred_lbs = logit_pred.argmax(dim=2)
        true_lbs = minibatch_y.argmax(dim=2)

        diff = true_lbs.ne(pred_lbs)
        accuracy_test = 1 - diff.any(1).sum().item() / len(dataset)
        print(f"Accuracy model in test data: {accuracy_test:.4f}")


if __name__ == "__main__":
    model = _load_model(
        PATH_DATA / "task1-data.tsv", PATH_MODELS / "task1-model.pth"
    )
    _evaluate_model(PATH_TEST / "task1-data.tsv", model)

    one_sample = "ABBBCDA"
    vocab = {"A": 0, "B": 1, "C": 2, "D": 3}

    vector_sample = _text_to_one_hot(one_sample, vocab, 9).unsqueeze(0)
    logits, attention = model(vector_sample)
