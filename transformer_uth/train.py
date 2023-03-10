from pathlib import Path

import torch
from torch import optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from transformer_uth.attention_models import AttentionModel, TransformerModel
from transformer_uth.consts import PATH_DATA, PATH_MODELS
from transformer_uth.vectorizer import CharVectorizer


def _train_model(
    path_data: Path,
    transformer: TransformerModel,
    epochs: int = 3,
    batch_size: int = 200,
) -> TransformerModel:
    dataset = CharVectorizer(path_data)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    model = transformer(dataset.seq_data)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1000, verbose=True
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Starting training model with {n_params} parameters")
    print("Model description: ")
    print(model)

    print("Model parameters: ")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,}")

    for i in range(epochs):
        for batch in data_loader:
            minibatch_x, minibatch_y = batch
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out, *_ = model(minibatch_x)
                loss = cross_entropy(
                    out.transpose(1, 2), minibatch_y.argmax(dim=2)
                )
                loss.backward()
                optimizer.step()
                lr_scheduler.step(loss)
        print(f"Iteration {i+1} - Loss {loss}")
    print("Training complete!")

    return model


def _save_model(model: TransformerModel, name_model: str) -> None:
    PATH_MODELS.mkdir(parents=True, exist_ok=True)
    path_model = PATH_MODELS / name_model
    torch.save(model.state_dict(), path_model)
    print(f"Model stored in {path_model} ")


if __name__ == "__main__":
    name_task = "task1"
    path_dat = PATH_DATA / f"{name_task}-data.tsv"
    model = _train_model(path_dat, AttentionModel)

    _save_model(model, f"{name_task}-model.pth")
