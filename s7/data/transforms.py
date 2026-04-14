"""Numpy event-stream augmentations used by the S7 dataset loaders.

All transforms operate on tonic structured-array events with fields
``(x, y, p, t)``. Each transform is callable ``events -> events``. Geometric
transforms remove events that fall outside the sensor after augmentation.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


SensorSize = Sequence[int]


class Identity:
    def __call__(self, events):
        return events


class Jitter1D:
    """Additive Gaussian jitter on the 1D coordinate ``x`` (used for audio-event streams).

    Drops events whose jittered position falls outside the sensor.
    """

    def __init__(self, sensor_size: SensorSize, var: float):
        self.sensor_size = sensor_size
        self.var = var

    def __call__(self, events):
        shift = np.random.normal(0, self.var, len(events)).astype(np.int32)
        events["x"] = events["x"] + shift
        m = (events["x"] >= 0) & (events["x"] < self.sensor_size[0])
        return events[m]


class Roll:
    """Translate (x, y) by a uniform random integer; drop out-of-frame events."""

    def __init__(self, sensor_size: SensorSize, p: float, max_roll: int):
        self.sx, self.sy = sensor_size[0], sensor_size[1]
        self.max_roll = max_roll
        self.p = p

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        events["x"] += np.random.randint(-self.max_roll, self.max_roll)
        events["y"] += np.random.randint(-self.max_roll, self.max_roll)
        m = (events["x"] >= 0) & (events["x"] < self.sx) & (events["y"] >= 0) & (events["y"] < self.sy)
        return events[m]


class Rotate:
    """Rotate (x, y) about the sensor centre by a uniform random angle."""

    def __init__(self, sensor_size: SensorSize, p: float, max_angle_deg: float):
        self.p = p
        self.sx, self.sy = sensor_size[0], sensor_size[1]
        self.max_angle = 2 * np.pi * max_angle_deg / 360.0

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        cx, cy = self.sx / 2, self.sy / 2
        x = events["x"] - cx
        y = events["y"] - cy
        c, s = np.cos(angle), np.sin(angle)
        events["x"] = np.clip((x * c - y * s + cx).astype(np.int32), 0, self.sx - 1)
        events["y"] = np.clip((x * s + y * c + cy).astype(np.int32), 0, self.sy - 1)
        return events


class Scale:
    """Scale (x, y) about the sensor centre by a uniform random factor in [1/s, s]."""

    def __init__(self, sensor_size: SensorSize, p: float, max_scale: float):
        assert max_scale >= 1
        self.p = p
        self.sx, self.sy = sensor_size[0], sensor_size[1]
        self.max_scale = max_scale

    def __call__(self, events):
        if np.random.rand() > self.p:
            return events
        s = np.random.uniform(1 / self.max_scale, self.max_scale)
        cx, cy = self.sx / 2, self.sy / 2
        events["x"] = ((events["x"] - cx) * s + cx).astype(np.int32)
        events["y"] = ((events["y"] - cy) * s + cy).astype(np.int32)
        m = (events["x"] >= 0) & (events["x"] < self.sx) & (events["y"] >= 0) & (events["y"] < self.sy)
        return events[m]


class DropEventChunk:
    """With probability ``p``, delete a contiguous chunk of up to ``max_drop_size·N`` events."""

    def __init__(self, p: float, max_drop_size: float):
        self.p = p
        self.max_drop_size = max_drop_size

    def __call__(self, events):
        if np.random.rand() >= self.p:
            return events
        max_drop = max(1, int(self.max_drop_size * len(events)))
        drop_size = np.random.randint(1, max_drop + 1)
        if drop_size >= len(events):
            return events
        start = np.random.randint(0, len(events) - drop_size)
        return np.delete(events, slice(start, start + drop_size), axis=0)


class OneHotLabels:
    def __init__(self, num_classes: int):
        self.eye = np.eye(num_classes, dtype=np.float32)

    def __call__(self, label):
        return self.eye[label]


# ---------------------------------------------------------------------------
# CutMix for variable-length event streams (time-domain)
# ---------------------------------------------------------------------------

def cut_mix_time(events: Sequence[np.ndarray], targets: Sequence[np.ndarray]
                 ) -> Tuple[list, list]:
    """Time-domain CutMix: for each item ``i`` swap a random time slice with item ``π(i)``.

    The mixed target is a convex combination weighted by the duration of the swapped slice
    relative to the resulting stream's total duration.
    """
    durations = np.array([e["t"][-1] - e["t"][0] for e in events], dtype=np.float32)
    cut_size = np.random.uniform(0, durations)
    cut_start = np.random.uniform(0, durations - cut_size)
    perm = np.random.permutation(len(events))

    out_events, out_targets = [], []
    for i in range(len(events)):
        s, e_t = cut_start[perm[i]], cut_start[perm[i]] + cut_size[perm[i]]
        mask_a = (events[i]["t"] >= s) & (events[i]["t"] <= e_t)
        mask_b = (events[perm[i]]["t"] >= s) & (events[perm[i]]["t"] <= e_t)

        merged = np.concatenate([events[i][~mask_a], events[perm[i]][mask_b]])
        if len(merged) == 0:
            out_events.append(events[i])
            out_targets.append(targets[i])
            continue

        merged = merged[np.argsort(merged["t"])]
        out_events.append(merged)
        merged_dur = merged["t"][-1] - merged["t"][0]
        chunk = events[perm[i]]["t"][mask_b]
        chunk_dur = (chunk[-1] - chunk[0]) if len(chunk) > 0 else 0.0
        lam = float(chunk_dur) / float(merged_dur) if merged_dur > 0 else 0.0
        lam = float(np.clip(lam, 0.0, 1.0))
        out_targets.append(targets[i] * (1.0 - lam) + targets[perm[i]] * lam)
    return out_events, out_targets
