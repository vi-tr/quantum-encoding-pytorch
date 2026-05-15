from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    macro_recall: float


@dataclass(frozen=True)
class FitResult:
    best_state: dict[str, Tensor]
    train_time_seconds: float
    best_val: EpochMetrics
    history: list[EpochMetrics]


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    macro_recall: float
    balanced_accuracy: float
    auroc: float | None


def make_loader(x: Tensor, y: Tensor, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def _collect_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_batches: list[Tensor] = []
    target_batches: list[Tensor] = []
    with torch.inference_mode():
        for x, y in loader:
            logits_batches.append(model(x.to(device)).cpu())
            target_batches.append(y.cpu())
    return torch.cat(logits_batches).numpy(), torch.cat(target_batches).numpy()


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> EvaluationResult:
    logits, targets = _collect_logits(model, loader, device)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = probs.argmax(axis=1)

    loss = float(
        nn.functional.cross_entropy(
            torch.from_numpy(logits), torch.from_numpy(targets)
        ).item()
    )
    accuracy = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro"))
    macro_recall = float(recall_score(targets, preds, average="macro"))
    balanced_accuracy = macro_recall

    auroc: float | None = None
    if len(np.unique(targets)) == 2:
        try:
            auroc = float(roc_auc_score(targets, probs[:, 1]))
        except ValueError:
            auroc = None

    return EvaluationResult(
        loss=loss,
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_recall=macro_recall,
        balanced_accuracy=balanced_accuracy,
        auroc=auroc,
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
) -> FitResult:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val = EpochMetrics(
        loss=float("inf"), accuracy=0.0, macro_f1=0.0, macro_recall=0.0
    )
    best_score = -float("inf")
    history: list[EpochMetrics] = []
    epochs_without_improvement = 0

    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_eval = evaluate_model(model, val_loader, device=device)
        epoch_metrics = EpochMetrics(
            loss=val_eval.loss,
            accuracy=val_eval.accuracy,
            macro_f1=val_eval.macro_f1,
            macro_recall=val_eval.macro_recall,
        )
        history.append(epoch_metrics)

        if epoch_metrics.macro_f1 > best_score:
            best_score = epoch_metrics.macro_f1
            best_val = epoch_metrics
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    train_time_seconds = time.perf_counter() - started
    return FitResult(
        best_state=best_state,
        train_time_seconds=train_time_seconds,
        best_val=best_val,
        history=history,
    )
