"""
Transformer adaptation from:
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

from pathlib import Path
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

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
    create_mask,
    PAD_IDX,
    Seq2SeqDataset,
)

BATCH_SIZE = 5
NUM_EPOCHS = 100


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, optimizer, train_dataset, loss_fn):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    for src_ids, tgt_ids in train_dataloader:
        src = src_ids.T
        tgt = tgt_ids.T

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        # tgt_input = tgt

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
        )
        loss.backward()

        optimizer.step()

        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model, valid_dataset, loss_fn):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    for src, tgt in val_dataloader:
        src = src.T.to(DEVICE)
        tgt = tgt.T.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
        )
        losses += loss.item()

    return losses / len(val_dataloader)


def _train_seq_transformer_model(
    path_train_data: Path, path_test_data: Path
) -> Seq2SeqTransformer:
    torch.manual_seed(0)
    train_dataset = Seq2SeqDataset(path_train_data)
    test_dataset = Seq2SeqDataset(path_test_data)

    transformer = Seq2SeqTransformer(
        len(train_dataset.input_vocab),
        len(train_dataset.output_vocab),
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        EMB_SIZE,
        NHEAD,
        FFN_HID_DIM,
    )

    # for model_params in transformer.parameters():
    #     if model_params.dim() > 1:
    #         nn.init.xavier_uniform_(model_params)

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(
            transformer, optimizer, train_dataset, loss_fn
        )

        end_time = timer()

        val_loss = evaluate(transformer, test_dataset, loss_fn)
        print(
            (
                f"Epoch: {epoch}, Train loss: {train_loss:.3f},"
                f" Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"
            )
        )

    return transformer


def _save_model(model: Seq2SeqTransformer, name_model: str) -> None:
    PATH_MODELS.mkdir(parents=True, exist_ok=True)
    path_model = PATH_MODELS / name_model
    torch.save(model.state_dict(), path_model)
    print(f"Model stored in {path_model} ")


if __name__ == "__main__":
    task = "es-numbers-translation-0-221"
    train = PATH_DATA / f"{task}.tsv"
    test = PATH_DATA / "es-numbers-translation-221-241.tsv"
    model = _train_seq_transformer_model(train, test)
    _save_model(model, f"{task}.pth")
