from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import torch

from .data import DATASETS, DatasetBundle
from .models import (
    PlainMLPClassifier,
    QuantumFeatureClassifier,
    ReservoirFeatureClassifier,
)
from .train import evaluate_model, fit_model, make_loader


@dataclass(frozen=True)
class BenchmarkConfig:
    datasets: tuple[str, ...] = ("moons", "breast_cancer", "digits", "creditcard")
    seeds: tuple[int, ...] = (0, 1, 2)
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    patience: int = 5
    hidden_dim: int = 128
    reservoir_dim: int = 64
    output_dir: str = "reports"


@dataclass(frozen=True)
class BenchmarkResult:
    dataset: str
    model: str
    seed: int
    train_time_seconds: float
    inference_time_per_sample_ms: float
    train_accuracy: float
    train_macro_f1: float
    train_macro_recall: float
    val_accuracy: float
    val_macro_f1: float
    val_macro_recall: float
    test_accuracy: float
    test_macro_f1: float
    test_macro_recall: float
    test_balanced_accuracy: float
    test_auroc: float | None
    num_parameters: int


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()

    def _device(self) -> torch.device:
        return torch.device("cpu")

    def _load_dataset(self, name: str, seed: int) -> DatasetBundle:
        spec = DATASETS[name]
        return spec.loader(seed)

    def _build_models(
        self, bundle: DatasetBundle, seed: int
    ) -> dict[str, torch.nn.Module]:
        return {
            "ann": PlainMLPClassifier(
                input_dim=bundle.input_dim,
                num_classes=bundle.num_classes,
                hidden_dim=self.config.hidden_dim,
            ),
            "quantum": QuantumFeatureClassifier(
                input_dim=bundle.input_dim,
                num_classes=bundle.num_classes,
                hidden_dim=self.config.hidden_dim,
            ),
            "reservoir": ReservoirFeatureClassifier(
                input_dim=bundle.input_dim,
                num_classes=bundle.num_classes,
                hidden_dim=self.config.hidden_dim,
                reservoir_dim=self.config.reservoir_dim,
                seed=seed,
            ),
        }

    def run(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        device = self._device()

        for dataset_name in self.config.datasets:
            for seed in self.config.seeds:
                torch.manual_seed(seed)
                bundle = self._load_dataset(dataset_name, seed)
                train_loader = make_loader(
                    bundle.x_train,
                    bundle.y_train,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                )
                val_loader = make_loader(
                    bundle.x_val,
                    bundle.y_val,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                )
                test_loader = make_loader(
                    bundle.x_test,
                    bundle.y_test,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                )

                for model_name, model in self._build_models(bundle, seed).items():
                    fit_result = fit_model(
                        model,
                        train_loader,
                        val_loader,
                        device=device,
                        epochs=self.config.epochs,
                        lr=self.config.learning_rate,
                        patience=self.config.patience,
                    )
                    model.load_state_dict(fit_result.best_state)

                    train_eval = evaluate_model(model, train_loader, device=device)
                    val_eval = fit_result.best_val
                    test_eval = evaluate_model(model, test_loader, device=device)

                    with torch.inference_mode():
                        for _ in range(3):
                            for x, _ in test_loader:
                                _ = model(x.to(device))
                    inference_started = perf_counter()
                    with torch.inference_mode():
                        for _ in range(5):
                            for x, _ in test_loader:
                                _ = model(x.to(device))
                    inference_elapsed = perf_counter() - inference_started
                    inference_time_per_sample_ms = (
                        inference_elapsed / (5 * len(bundle.x_test))
                    ) * 1000.0

                    results.append(
                        BenchmarkResult(
                            dataset=dataset_name,
                            model=model_name,
                            seed=seed,
                            train_time_seconds=fit_result.train_time_seconds,
                            inference_time_per_sample_ms=inference_time_per_sample_ms,
                            train_accuracy=train_eval.accuracy,
                            train_macro_f1=train_eval.macro_f1,
                            train_macro_recall=train_eval.macro_recall,
                            val_accuracy=val_eval.accuracy,
                            val_macro_f1=val_eval.macro_f1,
                            val_macro_recall=val_eval.macro_recall,
                            test_accuracy=test_eval.accuracy,
                            test_macro_f1=test_eval.macro_f1,
                            test_macro_recall=test_eval.macro_recall,
                            test_balanced_accuracy=test_eval.balanced_accuracy,
                            test_auroc=test_eval.auroc,
                            num_parameters=sum(
                                p.numel() for p in model.parameters() if p.requires_grad
                            ),
                        )
                    )

        return results


def summarize_results(results: list[BenchmarkResult]) -> dict[str, object]:
    return {"results": [asdict(result) for result in results]}
