"""DVS128 Gesture (IBM) data loader for S7.

Wraps tonic's preprocessed ``DVSGesture`` dataset with the augmentations and
collate function used by the S7 paper. 11 classes, 1077 train / 264 test
recordings at 128×128×2 (polarity).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import tonic
import torch
from torch.utils.data import DataLoader, Subset

from s7.data import DatasetInfo
from s7.data.collate import event_stream_collate_fn
from s7.data.transforms import (
    DropEventChunk,
    Identity,
    OneHotLabels,
    Roll,
    Rotate,
    Scale,
)


DVS_NUM_CLASSES = 11
DVS_SENSOR_SIZE = (128, 128, 2)


def _build_train_transforms(
    sensor_size,
    *,
    drop_event: float,
    noise: int,
    time_skew: float,
    time_jitter: float,
    spatial_jitter: float,
    max_drop_chunk: float,
    max_roll: int,
    max_angle: float,
    max_scale: float,
    downsample_to,
):
    return tonic.transforms.Compose([
        DropEventChunk(p=0.3, max_drop_size=max_drop_chunk),
        tonic.transforms.DropEvent(p=drop_event),
        tonic.transforms.UniformNoise(sensor_size=sensor_size, n=(0, noise)) if noise > 0 else Identity(),
        tonic.transforms.TimeSkew(coefficient=(1 / time_skew, time_skew), offset=0),
        tonic.transforms.TimeJitter(std=time_jitter, clip_negative=False, sort_timestamps=True),
        tonic.transforms.SpatialJitter(
            sensor_size=DVS_SENSOR_SIZE,
            var_x=spatial_jitter, var_y=spatial_jitter, clip_outliers=True,
        ),
        tonic.transforms.Downsample(sensor_size=DVS_SENSOR_SIZE, target_size=downsample_to)
            if downsample_to != DVS_SENSOR_SIZE[:2] else Identity(),
        Roll(sensor_size=sensor_size, p=0.3, max_roll=max_roll),
        Rotate(sensor_size=sensor_size, p=0.3, max_angle_deg=max_angle),
        Scale(sensor_size=sensor_size, p=0.3, max_scale=max_scale),
    ])


def build_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    slice_events: int = 0,
    pad_unit: int = 2 ** 19,
    time_jitter: float = 100,
    spatial_jitter: float = 1.0,
    noise: int = 100,
    drop_event: float = 0.1,
    time_skew: float = 1.1,
    cut_mix: float = 0.5,
    downsampling: int = 1,
    max_roll: int = 4,
    max_angle: float = 10,
    max_scale: float = 1.5,
    max_drop_chunk: float = 0.1,
    validate_on_test: bool = False,
    slice_val_set: bool = False,
    **_unused,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    """Build train/val/test :class:`DataLoader`s for DVS128 Gesture.

    Many of the kwargs come straight from the task YAML; unknown extras are
    ignored so configs can carry SSM-side keys via ``**cfg.training,
    **cfg.model.ssm`` without breaking this builder.
    """
    assert time_skew > 1, "time_skew must be > 1 (it's used as the half-width of a log-uniform window)"
    # Tonic creates its own "DVSGesture/" subdir under save_to.
    cache_dir = Path(cache_dir) / "dvs128"

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    sensor_size = (128 // downsampling, 128 // downsampling, 2)
    downsample_to = sensor_size[:2]

    train_tf = _build_train_transforms(
        sensor_size,
        drop_event=drop_event, noise=noise, time_skew=time_skew,
        time_jitter=time_jitter, spatial_jitter=spatial_jitter,
        max_drop_chunk=max_drop_chunk, max_roll=max_roll,
        max_angle=max_angle, max_scale=max_scale,
        downsample_to=downsample_to,
    )
    test_tf = tonic.transforms.Compose([
        tonic.transforms.Downsample(sensor_size=DVS_SENSOR_SIZE, target_size=downsample_to)
            if downsample_to != DVS_SENSOR_SIZE[:2] else Identity(),
    ])
    target_tf = OneHotLabels(num_classes=DVS_NUM_CLASSES)

    Train = partial(tonic.datasets.DVSGesture, save_to=str(cache_dir), train=True)
    Test = partial(tonic.datasets.DVSGesture, save_to=str(cache_dir), train=False)

    if validate_on_test:
        val_data = Test(transform=test_tf, target_transform=target_tf)
    else:
        full_train = Train(transform=test_tf, target_transform=target_tf)
        n_val = int(0.2 * len(full_train))
        idx = torch.randperm(len(full_train), generator=rng)
        val_data = Subset(full_train, idx[-n_val:])

    if slice_events > 0:
        slicer = tonic.slicers.SliceByEventCount(
            event_count=slice_events, overlap=slice_events // 2, include_incomplete=True,
        )
        train_data = tonic.SlicedDataset(
            Train(),
            slicer=slicer,
            transform=train_tf,
            target_transform=target_tf,
            metadata_path=str(cache_dir / "train_slice_metadata"),
        )
        if slice_val_set:
            val_data = tonic.SlicedDataset(
                Test() if validate_on_test else Train(),
                slicer=slicer,
                transform=test_tf,
                target_transform=target_tf,
                metadata_path=str(cache_dir / "val_slice_metadata"),
            )
    else:
        train_data = Train(transform=train_tf, target_transform=target_tf)

    test_data = Test(transform=test_tf, target_transform=target_tf)

    collate_train = partial(
        event_stream_collate_fn,
        resolution=sensor_size[:2], pad_unit=pad_unit, cut_mix=cut_mix,
    )
    collate_eval = partial(
        event_stream_collate_fn,
        resolution=sensor_size[:2], pad_unit=pad_unit, cut_mix=0.0,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=per_device_batch_size * world_size,
        shuffle=True, drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_train,
        generator=rng,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=per_device_eval_batch_size * world_size,
        shuffle=False, drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_eval,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=per_device_eval_batch_size * world_size,
        shuffle=False, drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_eval,
        persistent_workers=num_workers > 0,
    )

    info = DatasetInfo(
        n_classes=DVS_NUM_CLASSES,
        num_embeddings=int(np.prod(sensor_size)),
        train_size=len(train_data),
    )
    return train_loader, val_loader, test_loader, info
