"""Irregularly-sampled physical / biological time-series loaders.

- **EigenWorms** — 5-way worm behaviour classification, 6-dimensional vector
  observations. Data lives as pre-processed ``.pt`` files under
  ``<cache_dir_parent>/eigenworms/processed/``.
- **PersonActivity** — 7-class activity recognition, 7-dimensional sensor
  vectors with irregular timestamps. Wraps ``odelstms.irregular_sampled_datasets.PersonData``.
- **Walker2D** — 17-dimensional imitation sequences. Wraps
  ``odelstms.irregular_sampled_datasets.Walker2dImitationData``.

Keeping the ``odelstms/`` tree as a vendored data-prep dependency (just like
``S5/``) is intentional — these wrappers unify the interface without touching
the underlying data code.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from tonic.io import make_structured_array
from torch.utils.data import DataLoader, Dataset

from s7.data import DatasetInfo
from s7.data.collate import eigenworms_collate_fn, irregular_collate_fn
from s7.data.transforms import OneHotLabels


def _make_loader(ds, bsz, collate, *, shuffle, drop_last, rng, num_workers):
    return DataLoader(
        ds,
        batch_size=bsz,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate,
        num_workers=num_workers,
        generator=rng,
        persistent_workers=num_workers > 0,
    )


def _rng(seed):
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# EigenWorms
# ---------------------------------------------------------------------------

class _EigenWormsDataset(Dataset):
    """Loads ``{training,validation,test}.pt`` files produced by the LEM repo's
    eigenworms preprocessing pipeline."""

    _DTYPE = np.dtype([("t", int), ("x", (float, 6)), ("p", int)])

    def __init__(self, data_dir: Path, split: str, target_transform):
        self.target_transform = target_transform
        fname = {"train": "training.pt", "val": "validation.pt", "test": "test.pt"}[split]
        self.data = torch.load(data_dir / fname)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series, target = self.data[idx]
        times = np.arange(len(series))
        events = make_structured_array(times, series, 1, dtype=self._DTYPE)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target


def build_eigenworms_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 2048,
    cut_mix: float = 0.0,
    no_time_information: bool = False,
    **_unused,
):
    rng = _rng(seed)
    data_dir = Path(cache_dir) / "eigenworms/processed"
    tgt_tf = OneHotLabels(num_classes=5)

    train = _EigenWormsDataset(data_dir, "train", tgt_tf)
    val = _EigenWormsDataset(data_dir, "val", tgt_tf)
    test = _EigenWormsDataset(data_dir, "test", tgt_tf)

    collate = partial(
        eigenworms_collate_fn, resolution=(6,), pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
    train_loader = _make_loader(train, per_device_batch_size * world_size, collate,
                                shuffle=True, drop_last=True, rng=rng, num_workers=num_workers)
    val_loader = _make_loader(val, per_device_eval_batch_size * world_size, collate,
                              shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    test_loader = _make_loader(test, per_device_eval_batch_size * world_size, collate,
                               shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    info = DatasetInfo(n_classes=5, num_embeddings=6, train_size=len(train))
    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------------------------
# PersonActivity (ConfLongDemo)
# ---------------------------------------------------------------------------

class _PersonActivityDataset(Dataset):
    _DTYPE = np.dtype([("t", float), ("x", (float, 7)), ("p", int)])

    def __init__(self, data_path: Path, split: str, target_transform):
        from odelstms.irregular_sampled_datasets import PersonData
        self.target_transform = target_transform
        self._pd = PersonData(data_dir=str(data_path))
        if split == "train":
            self.data = self._pd.train_x
            self.times = self._pd.train_t.squeeze()
            self.target = self._pd.train_y
        else:
            self.data = self._pd.test_x
            self.times = self._pd.test_t.squeeze()
            self.target = self._pd.test_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series = self.data[idx]
        times = self.times[idx]
        target = self.target[idx]
        events = make_structured_array(times, series, 1, dtype=self._DTYPE)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target


def build_personactivity_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 2048,
    cut_mix: float = 0.0,
    no_time_information: bool = False,
    **_unused,
):
    rng = _rng(seed)
    data_path = Path(cache_dir) / "person/ConfLongDemo_JSI.txt"
    tgt_tf = OneHotLabels(num_classes=7)

    train = _PersonActivityDataset(data_path, "train", tgt_tf)
    # PersonData exposes only train/test, so val = test.
    val = _PersonActivityDataset(data_path, "test", tgt_tf)
    test = _PersonActivityDataset(data_path, "test", tgt_tf)

    collate = partial(
        irregular_collate_fn, resolution=(7,), pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
    train_loader = _make_loader(train, per_device_batch_size * world_size, collate,
                                shuffle=True, drop_last=True, rng=rng, num_workers=num_workers)
    val_loader = _make_loader(val, per_device_eval_batch_size * world_size, collate,
                              shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    test_loader = _make_loader(test, per_device_eval_batch_size * world_size, collate,
                               shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    info = DatasetInfo(n_classes=7, num_embeddings=7, train_size=len(train))
    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------------------------
# Walker2D imitation
# ---------------------------------------------------------------------------

class _WalkerDataset(Dataset):
    _DTYPE = np.dtype([("t", float), ("x", (float, 17)), ("p", int)])

    def __init__(self, data_dir: Path, split: str):
        from odelstms.irregular_sampled_datasets import Walker2dImitationData
        self._wd = Walker2dImitationData(data_dir=str(data_dir), seq_len=64)
        if split == "train":
            self.data = self._wd.train_x
            self.times = self._wd.train_times.squeeze()
            self.target = self._wd.train_y
        else:
            self.data = self._wd.test_x
            self.times = self._wd.test_times.squeeze()
            self.target = self._wd.test_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series = self.data[idx]
        times = self.times[idx]
        target = self.target[idx]
        events = make_structured_array(times, series, 1, dtype=self._DTYPE)
        return events, target


def build_walker_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 2048,
    cut_mix: float = 0.0,
    no_time_information: bool = False,
    **_unused,
):
    rng = _rng(seed)
    data_dir = Path(cache_dir) / "walker/"

    train = _WalkerDataset(data_dir, "train")
    val = _WalkerDataset(data_dir, "test")
    test = _WalkerDataset(data_dir, "test")

    collate = partial(
        irregular_collate_fn, resolution=(17,), pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
    train_loader = _make_loader(train, per_device_batch_size * world_size, collate,
                                shuffle=True, drop_last=True, rng=rng, num_workers=num_workers)
    val_loader = _make_loader(val, per_device_eval_batch_size * world_size, collate,
                              shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    test_loader = _make_loader(test, per_device_eval_batch_size * world_size, collate,
                               shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    info = DatasetInfo(n_classes=17, num_embeddings=17, train_size=len(train))
    return train_loader, val_loader, test_loader, info
