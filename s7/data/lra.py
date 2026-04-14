"""Long Range Arena (LRA) loaders: ListOps, Text (IMDB), Retrieval (AAN),
Image (CIFAR10-grayscale), Pathfinder, Path-X.

These wrap the vendored ``S5/s5/dataloaders/{lra,basic}`` data-prep code, which
handles downloading the LRA tarball, tokenization, and train/val/test splits.
We keep ``S5/`` as an external data-prep dependency (like tonic) but route
everything through the new ``s7.data`` registry so the S7 training code never
touches ``event_ssm``.

The S5 dataloaders expect the LRA data already unpacked under
``<cache_dir_parent>/long-range-arena/...``. The ``lra_release.gz`` tarball is
available from the LRA GitHub release page.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tonic.io import make_structured_array
from torch.utils.data import DataLoader, Dataset

from s7.data import DatasetInfo
from s7.data.collate import (
    event_stream_collate_fn,
    lra_image_collate_fn,
    lra_text_collate_fn,
    retrieval_collate_fn,
)
from s7.data.transforms import OneHotLabels


# The vendored S5 LRA dataloaders live under S5/s5/dataloaders. They're
# imported lazily (inside each dataset class) so that simply importing
# ``s7.data`` doesn't force the S5 tree to be on PYTHONPATH until the user
# actually asks for an LRA task.

_TOKEN_DTYPE = np.dtype([("t", int), ("x", int), ("p", int)])
_IMAGE_DTYPE = np.dtype([("t", int), ("x", int), ("p", int)])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _rng_from_seed(seed: Optional[int]) -> torch.Generator:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# ListOps (LRA)
# ---------------------------------------------------------------------------

class _ListOpsDataset(Dataset):
    """Tokenized ListOps sequences from the S5 LRA dataloader."""

    def __init__(self, data_dir, cache_dir, split, target_transform):
        from S5.s5.dataloaders.lra import ListOps
        self.split = split
        self.target_transform = target_transform
        self._lo = ListOps(
            _name_="listops", data_dir=data_dir, cache_dir=cache_dir,
            train=(split == "train"),
        )
        self._lo.prepare_data()
        self._lo.setup(stage=split)
        self.data = {
            "train": self._lo.dataset_train,
            "val": self._lo.dataset_val,
            "test": self._lo.dataset_test,
        }[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["input_ids"]
        target = item["Target"]
        times = np.arange(len(tokens))
        events = make_structured_array(times, tokens, 1, dtype=_TOKEN_DTYPE)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target


def build_listops_loaders(
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
    rng = _rng_from_seed(seed)
    data_dir = Path(cache_dir).parent / "long-range-arena/lra_release/lra_release/listops-1000"
    cache = data_dir / "l_max-2048-append_bos-False-append_eos-True"
    tgt_tf = OneHotLabels(num_classes=10)

    train = _ListOpsDataset(data_dir, cache, "train", tgt_tf)
    val = _ListOpsDataset(data_dir, cache, "val", tgt_tf)
    test = _ListOpsDataset(data_dir, cache, "test", tgt_tf)

    collate = partial(
        lra_text_collate_fn, resolution=(18,), pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
    train_loader = _make_loader(
        train, per_device_batch_size * world_size, collate,
        shuffle=True, drop_last=True, rng=rng, num_workers=num_workers,
    )
    val_loader = _make_loader(
        val, per_device_eval_batch_size * world_size, collate,
        shuffle=False, drop_last=False, rng=rng, num_workers=num_workers,
    )
    test_loader = _make_loader(
        test, per_device_eval_batch_size * world_size, collate,
        shuffle=False, drop_last=False, rng=rng, num_workers=num_workers,
    )
    info = DatasetInfo(n_classes=10, num_embeddings=18, train_size=len(train))
    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------------------------
# Text / IMDB (LRA)
# ---------------------------------------------------------------------------

class _IMDBDataset(Dataset):
    def __init__(self, data_dir, cache_dir, split, target_transform):
        from S5.s5.dataloaders.lra import IMDB
        self.split = split
        self.target_transform = target_transform
        self._imdb = IMDB(_name_="imdb", data_dir=data_dir, cache_dir=cache_dir)
        self._imdb.prepare_data()
        self._imdb.setup(stage=split)
        self.data = (
            self._imdb.dataset_train if split == "train" else self._imdb.dataset_test
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["input_ids"]
        label = item["label"]
        times = np.arange(len(tokens))
        events = make_structured_array(times, tokens, 1, dtype=_TOKEN_DTYPE)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return events, label


def build_text_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 4096,
    cut_mix: float = 0.0,
    no_time_information: bool = False,
    **_unused,
):
    rng = _rng_from_seed(seed)
    data_dir = Path(cache_dir).parent / "long-range-arena/text/"
    tgt_tf = OneHotLabels(num_classes=2)

    train = _IMDBDataset(data_dir, data_dir, "train", tgt_tf)
    test = _IMDBDataset(data_dir, data_dir, "test", tgt_tf)

    collate = partial(
        lra_text_collate_fn, resolution=(135,), pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
    train_loader = _make_loader(train, per_device_batch_size * world_size, collate,
                                shuffle=True, drop_last=True, rng=rng, num_workers=num_workers)
    val_loader = _make_loader(test, per_device_eval_batch_size * world_size, collate,
                              shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    test_loader = _make_loader(test, per_device_eval_batch_size * world_size, collate,
                               shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    info = DatasetInfo(n_classes=2, num_embeddings=135, train_size=len(train))
    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------------------------
# Retrieval / AAN (LRA)
# ---------------------------------------------------------------------------

class _AANDataset(Dataset):
    def __init__(self, data_dir, cache_dir, split, target_transform):
        from S5.s5.dataloaders.lra import AAN
        self.split = split
        self.target_transform = target_transform
        self._aan = AAN(
            _name_="aan", data_dir=data_dir, cache_dir=cache_dir,
            train=(split == "train"),
        )
        self._aan.prepare_data()
        self._aan.setup(stage=split)
        self.data = {
            "train": self._aan.dataset_train,
            "val": self._aan.dataset_val,
            "test": self._aan.dataset_test,
        }[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tok_a, tok_b = item["input_ids1"], item["input_ids2"]
        label = item["label"]
        t_a = np.arange(len(tok_a))
        t_b = np.arange(len(tok_b))
        e_a = make_structured_array(t_a, tok_a, 1, dtype=_TOKEN_DTYPE)
        e_b = make_structured_array(t_b, tok_b, 1, dtype=_TOKEN_DTYPE)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return (e_a, e_b), label


def build_retrieval_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 4000,
    cut_mix: float = 0.0,
    no_time_information: bool = True,
    **_unused,
):
    rng = _rng_from_seed(seed)
    data_dir = Path(cache_dir).parent / "long-range-arena/retrieval/"
    tgt_tf = OneHotLabels(num_classes=2)

    train = _AANDataset(data_dir, data_dir, "train", tgt_tf)
    val = _AANDataset(data_dir, data_dir, "val", tgt_tf)
    test = _AANDataset(data_dir, data_dir, "test", tgt_tf)

    collate = partial(
        retrieval_collate_fn, resolution=(98,), pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
    train_loader = _make_loader(train, per_device_batch_size * world_size, collate,
                                shuffle=True, drop_last=True, rng=rng, num_workers=num_workers)
    val_loader = _make_loader(val, per_device_eval_batch_size * world_size, collate,
                              shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    test_loader = _make_loader(test, per_device_eval_batch_size * world_size, collate,
                               shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    info = DatasetInfo(n_classes=2, num_embeddings=98, train_size=len(train))
    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------------------------
# Image (LRA CIFAR-10 grayscale)
# ---------------------------------------------------------------------------

class _ImageDataset(Dataset):
    def __init__(self, data_dir, cache_dir, split, target_transform):
        from S5.s5.dataloaders.basic import CIFAR10
        self.split = split
        self.target_transform = target_transform
        self._img = CIFAR10(
            _name_="cifar", data_dir=data_dir, cache_dir=cache_dir,
            grayscale=True, tokenize=True,
        )
        self._img.setup()
        self.data = {
            "train": self._img.dataset_train,
            "val": self._img.dataset_val,
            "test": self._img.dataset_test,
        }[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, target = self.data[idx]
        tokens = np.squeeze(tokens)
        times = np.arange(len(tokens))
        events = make_structured_array(times, tokens, 1, dtype=_IMAGE_DTYPE)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target


def build_image_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 1024,
    cut_mix: float = 0.0,
    no_time_information: bool = False,
    **_unused,
):
    rng = _rng_from_seed(seed)
    data_dir = Path(cache_dir).parent / "long-range-arena/image/"
    tgt_tf = OneHotLabels(num_classes=10)

    train = _ImageDataset(data_dir, data_dir, "train", tgt_tf)
    val = _ImageDataset(data_dir, data_dir, "val", tgt_tf)
    test = _ImageDataset(data_dir, data_dir, "test", tgt_tf)

    collate = partial(
        lra_image_collate_fn, resolution=(256,), pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
    train_loader = _make_loader(train, per_device_batch_size * world_size, collate,
                                shuffle=True, drop_last=True, rng=rng, num_workers=num_workers)
    val_loader = _make_loader(val, per_device_eval_batch_size * world_size, collate,
                              shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    test_loader = _make_loader(test, per_device_eval_batch_size * world_size, collate,
                               shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    info = DatasetInfo(n_classes=10, num_embeddings=256, train_size=len(train))
    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------------------------
# Pathfinder / Path-X
# ---------------------------------------------------------------------------

class _PathFinderDataset(Dataset):
    def __init__(self, data_dir, cache_dir, split, resolution, target_transform):
        from S5.s5.dataloaders.lra import PathFinder
        self.split = split
        self.target_transform = target_transform
        self._pf = PathFinder(
            _name_="pathfinder", data_dir=data_dir, cache_dir=cache_dir,
            resolution=resolution, tokenize=True,
        )
        self._pf.setup()
        self.data = {
            "train": self._pf.dataset_train,
            "val": self._pf.dataset_val,
            "test": self._pf.dataset_test,
        }[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, target = self.data[idx]
        tokens = np.squeeze(tokens)
        times = np.arange(len(tokens))
        events = make_structured_array(times, tokens, 1, dtype=_IMAGE_DTYPE)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target


def _build_pathfinder_family(
    *,
    resolution: int,
    cache_dir: str,
    per_device_batch_size: int,
    per_device_eval_batch_size: int,
    world_size: int,
    num_workers: int,
    seed: int,
    pad_unit: int,
    no_time_information: bool,
):
    rng = _rng_from_seed(seed)
    data_dir = Path(cache_dir).parent / "long-range-arena/pathfinder/"
    tgt_tf = OneHotLabels(num_classes=2)

    train = _PathFinderDataset(data_dir, data_dir, "train", resolution, tgt_tf)
    val = _PathFinderDataset(data_dir, data_dir, "val", resolution, tgt_tf)
    test = _PathFinderDataset(data_dir, data_dir, "test", resolution, tgt_tf)

    collate = partial(
        lra_image_collate_fn, resolution=(256,), pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
    train_loader = _make_loader(train, per_device_batch_size * world_size, collate,
                                shuffle=True, drop_last=True, rng=rng, num_workers=num_workers)
    val_loader = _make_loader(val, per_device_eval_batch_size * world_size, collate,
                              shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    test_loader = _make_loader(test, per_device_eval_batch_size * world_size, collate,
                               shuffle=False, drop_last=False, rng=rng, num_workers=num_workers)
    info = DatasetInfo(n_classes=2, num_embeddings=256, train_size=len(train))
    return train_loader, val_loader, test_loader, info


def build_pathfinder_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 1024,
    no_time_information: bool = True,
    resolution: int = 32,
    **_unused,
):
    return _build_pathfinder_family(
        resolution=resolution, cache_dir=cache_dir,
        per_device_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        world_size=world_size, num_workers=num_workers,
        seed=seed, pad_unit=pad_unit,
        no_time_information=no_time_information,
    )


def build_pathx_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 1024,
    no_time_information: bool = True,
    resolution: int = 128,
    **_unused,
):
    return _build_pathfinder_family(
        resolution=resolution, cache_dir=cache_dir,
        per_device_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        world_size=world_size, num_workers=num_workers,
        seed=seed, pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
