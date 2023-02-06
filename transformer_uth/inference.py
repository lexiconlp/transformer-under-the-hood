from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformer_uth.attention_models import AttentionModel, TransformerModel
from transformer_uth.consts import PATH_DATA, PATH_MODELS, PATH_TEST
from transformer_uth.vectorizer import _text_to_one_hot, CharVectorizer


def _load_model(
    vectorizer: CharVectorizer,
    path_model: Path,
    transformer_model: TransformerModel,
) -> TransformerModel:
    model = transformer_model(vectorizer.seq_data)
    model.load_state_dict(torch.load(path_model))
    return model


def _evaluate_model(path_test: Path, model: TransformerModel):
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
    name_task = "task1"
    path_dat = PATH_DATA / f"{name_task}-data.tsv"
    path_mod = PATH_MODELS / f"{name_task}-model.pth"
    vectorizer = CharVectorizer(path_dat)
    model = _load_model(vectorizer, path_mod, AttentionModel)

    _evaluate_model(PATH_TEST / f"{name_task}-data.tsv", model)

    one_sample = "ABBBCDA"
    vector_sample = _text_to_one_hot(
        one_sample, vectorizer.input_vocab, 9
    ).unsqueeze(0)
    logits, *attentions = model(vector_sample)
