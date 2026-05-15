from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml, load_breast_cancer, load_digits, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    input_dim: int
    num_classes: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable[[int], DatasetBundle]


def _to_torch(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64))


def _split_and_scale(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_dim: int | None,
    seed: int,
) -> DatasetBundle:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    if target_dim is not None and x_train.shape[1] > target_dim:
        pca = PCA(n_components=target_dim, svd_solver="randomized", random_state=seed)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)

    bounded_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    x_train = bounded_scaler.fit_transform(x_train)
    x_val = bounded_scaler.transform(x_val)
    x_test = bounded_scaler.transform(x_test)

    x_train_t, y_train_t = _to_torch(x_train, y_train)
    x_val_t, y_val_t = _to_torch(x_val, y_val)
    x_test_t, y_test_t = _to_torch(x_test, y_test)
    return DatasetBundle(
        name="",
        input_dim=x_train_t.shape[-1],
        num_classes=int(np.unique(y).size),
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        x_test=x_test_t,
        y_test=y_test_t,
    )


def load_moons(seed: int) -> DatasetBundle:
    x, y = make_moons(n_samples=4000, noise=0.25, random_state=seed)
    return replace(_split_and_scale(x, y, target_dim=2, seed=seed), name="moons")


def load_breast_cancer_dataset(seed: int) -> DatasetBundle:
    data = load_breast_cancer()
    return replace(
        _split_and_scale(data.data, data.target, target_dim=30, seed=seed),
        name="breast_cancer",
    )


def load_digits_dataset(seed: int) -> DatasetBundle:
    data = load_digits()
    return replace(
        _split_and_scale(data.data, data.target, target_dim=62, seed=seed),
        name="digits",
    )


@lru_cache(maxsize=1)
def _load_creditcard_csv() -> tuple[np.ndarray, np.ndarray]:
    path = Path("creditcard.csv")
    if not path.exists():
        raise FileNotFoundError(
            "creditcard.csv was not found at the project root. "
            "Place the dataset file there before running the benchmark."
        )

    frame = pd.read_csv(path)
    x = frame.drop(columns=["Class"]).to_numpy(dtype=np.float32)
    y = frame["Class"].to_numpy(dtype=np.int64)
    return x, y


def load_creditcard(seed: int) -> DatasetBundle:
    x, y = _load_creditcard_csv()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.arange(len(x)), size=50000, replace=False)
    x = x[chosen]
    y = y[chosen]
    return replace(_split_and_scale(x, y, target_dim=30, seed=seed), name="creditcard")


@lru_cache(maxsize=1)
def _load_mnist_openml() -> tuple[np.ndarray, np.ndarray]:
    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = data.data.astype(np.float32) / 255.0
    y = data.target.astype(np.int64)
    return x, y


def load_mnist_subset(seed: int) -> DatasetBundle:
    x, y = _load_mnist_openml()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.arange(len(x)), size=10000, replace=False)
    x = x[chosen]
    y = y[chosen]
    return replace(
        _split_and_scale(x, y, target_dim=510, seed=seed), name="mnist_subset"
    )


DATASETS: dict[str, DatasetSpec] = {
    "moons": DatasetSpec(name="moons", loader=load_moons),
    "breast_cancer": DatasetSpec(
        name="breast_cancer", loader=load_breast_cancer_dataset
    ),
    "digits": DatasetSpec(name="digits", loader=load_digits_dataset),
    "creditcard": DatasetSpec(name="creditcard", loader=load_creditcard),
    "mnist_subset": DatasetSpec(name="mnist_subset", loader=load_mnist_subset),
}
