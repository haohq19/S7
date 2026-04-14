"""Spiking Heidelberg Digits (SHD) and Spiking Speech Commands (SSC) loaders.

Both datasets are 1D audio-event streams with a 700-channel cochlea sensor and
the same augmentation/collate pattern; SHD has 20 classes and SSC has 35. We
share the implementation and expose two thin builders.

Tonic provides ``tonic.datasets.SHD`` and ``tonic.datasets.SSC`` which download
the preprocessed HDF5s automatically.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Tuple

import tonic
import torch
from torch.utils.data import DataLoader, Subset

from s7.data import DatasetInfo
from s7.data.collate import event_stream_collate_fn
from s7.data.transforms import DropEventChunk, Jitter1D, OneHotLabels


_SENSOR_SIZE = (700, 1, 1)  # 700-channel cochlea, 1 polarity, 1 dummy y-axis
_RESOLUTION = (700,)


def _build_transforms(
    *,
    drop_event: float,
    max_drop_chunk: float,
    spatial_jitter: float,
    time_skew: float,
    time_jitter: float,
    noise: int,
):
    return tonic.transforms.Compose([
        tonic.transforms.DropEvent(p=drop_event),
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        Jitter1D(sensor_size=_SENSOR_SIZE, var=spatial_jitter),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.UniformNoise(sensor_size=_SENSOR_SIZE, n=(0, noise)),
    ])


def _build_loaders_from_splits(
    *,
    train_data,
    val_data,
    test_data,
    per_device_batch_size: int,
    per_device_eval_batch_size: int,
    world_size: int,
    num_workers: int,
    rng: torch.Generator,
    pad_unit: int,
    cut_mix: float,
    no_time_information: bool,
    n_classes: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    collate_train = partial(
        event_stream_collate_fn,
        resolution=_RESOLUTION, pad_unit=pad_unit,
        cut_mix=cut_mix, no_time_information=no_time_information,
    )
    collate_eval = partial(
        event_stream_collate_fn,
        resolution=_RESOLUTION, pad_unit=pad_unit,
        cut_mix=0.0, no_time_information=no_time_information,
    )

    def _loader(ds, bsz, collate, shuffle, drop_last):
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

    train_loader = _loader(
        train_data, per_device_batch_size * world_size, collate_train,
        shuffle=True, drop_last=True,
    )
    val_loader = _loader(
        val_data, per_device_eval_batch_size * world_size, collate_eval,
        shuffle=False, drop_last=False,
    )
    test_loader = _loader(
        test_data, per_device_eval_batch_size * world_size, collate_eval,
        shuffle=False, drop_last=False,
    )
    info = DatasetInfo(
        n_classes=n_classes,
        num_embeddings=_SENSOR_SIZE[0],
        train_size=len(train_data),
    )
    return train_loader, val_loader, test_loader, info


def build_shd_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    time_jitter: float = 100,
    spatial_jitter: float = 1.0,
    max_drop_chunk: float = 0.1,
    noise: int = 100,
    drop_event: float = 0.1,
    time_skew: float = 1.1,
    cut_mix: float = 0.5,
    pad_unit: int = 8192,
    validate_on_test: bool = False,
    no_time_information: bool = False,
    **_unused,
):
    """Build loaders for SHD. Validation split = 10% of training unless ``validate_on_test``."""
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    # Tonic creates an "SHD/" subdir under save_to.
    cache_dir = str(Path(cache_dir) / "shd")

    train_tf = _build_transforms(
        drop_event=drop_event, max_drop_chunk=max_drop_chunk,
        spatial_jitter=spatial_jitter, time_skew=time_skew,
        time_jitter=time_jitter, noise=noise,
    )
    target_tf = OneHotLabels(num_classes=20)

    train_data = tonic.datasets.SHD(
        save_to=cache_dir, train=True,
        transform=train_tf, target_transform=target_tf,
    )
    test_data = tonic.datasets.SHD(
        save_to=cache_dir, train=False, target_transform=target_tf,
    )

    if validate_on_test:
        val_data = test_data
    else:
        # Pull a 10% slice of train *without* augmentation.
        train_no_aug = tonic.datasets.SHD(save_to=cache_dir, train=True, target_transform=target_tf)
        n_val = int(0.1 * len(train_data))
        idx = torch.randperm(len(train_data), generator=rng)
        train_data = Subset(train_data, idx[:-n_val])
        val_data = Subset(train_no_aug, idx[-n_val:])

    return _build_loaders_from_splits(
        train_data=train_data, val_data=val_data, test_data=test_data,
        per_device_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        world_size=world_size, num_workers=num_workers, rng=rng,
        pad_unit=pad_unit, cut_mix=cut_mix,
        no_time_information=no_time_information,
        n_classes=20,
    )


def build_ssc_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    time_jitter: float = 100,
    spatial_jitter: float = 1.0,
    max_drop_chunk: float = 0.1,
    noise: int = 100,
    drop_event: float = 0.1,
    time_skew: float = 1.1,
    cut_mix: float = 0.5,
    pad_unit: int = 8192,
    no_time_information: bool = False,
    **_unused,
):
    """Build loaders for SSC. SSC ships its own train/valid/test splits."""
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    # Tonic creates an "SSC/" subdir under save_to.
    cache_dir = str(Path(cache_dir) / "ssc")

    train_tf = _build_transforms(
        drop_event=drop_event, max_drop_chunk=max_drop_chunk,
        spatial_jitter=spatial_jitter, time_skew=time_skew,
        time_jitter=time_jitter, noise=noise,
    )
    target_tf = OneHotLabels(num_classes=35)

    train_data = tonic.datasets.SSC(
        save_to=cache_dir, split="train",
        transform=train_tf, target_transform=target_tf,
    )
    val_data = tonic.datasets.SSC(
        save_to=cache_dir, split="valid", target_transform=target_tf,
    )
    test_data = tonic.datasets.SSC(
        save_to=cache_dir, split="test", target_transform=target_tf,
    )

    return _build_loaders_from_splits(
        train_data=train_data, val_data=val_data, test_data=test_data,
        per_device_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        world_size=world_size, num_workers=num_workers, rng=rng,
        pad_unit=pad_unit, cut_mix=cut_mix,
        no_time_information=no_time_information,
        n_classes=35,
    )
