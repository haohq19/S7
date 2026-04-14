"""Collate function: turn variable-length event streams into a padded JAX-ready batch.

Output tuple: ``(tokens, targets, dt, lengths)``
    tokens   (B, L)     int32 — spatial-polarity tokens, padded with -1
    targets  (B, …)     stacked label arrays (one-hot or scalar)
    dt       (B, L)     float32 — Δt between consecutive events, in milliseconds
    lengths  (B,)       int32 — true event count per sample (excluding padding)

Padding length is rounded *up* to the next multiple of ``pad_unit`` so that
JAX only compiles a small, finite set of sequence-length kernels per epoch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from s7.data.transforms import cut_mix_time


def _tokenize_2d(events, sensor_w: int) -> np.ndarray:
    """Token = x · 2W + y · 2 + p — unique integer per (x, y, polarity) for a 2D sensor."""
    return (events["x"][:-1].astype(np.int32) * sensor_w * 2
            + events["y"][:-1].astype(np.int32) * 2
            + events["p"][:-1].astype(np.int32))


def _tokenize_1d(events) -> np.ndarray:
    return events["x"][:-1].astype(np.int32)


def event_stream_collate_fn(
    batch: Sequence,
    *,
    resolution: Sequence[int],
    pad_unit: int,
    cut_mix: float = 0.0,
    no_time_information: bool = False,
):
    """Collate variable-length event streams into a batched padded tensor.

    :param batch: iterable of ``(events, target)`` pairs from a tonic dataset.
    :param resolution: sensor resolution — len 1 (audio) or len 2 (DVS).
    :param pad_unit: pad each batch up to a multiple of this many timesteps.
    :param cut_mix: probability of applying time-domain CutMix to the batch.
    :param no_time_information: if True, set every Δt to 1 (ablation only).
    """
    inputs, targets = zip(*batch)

    if cut_mix > 0 and np.random.rand() < cut_mix:
        inputs, targets = cut_mix_time(inputs, targets)

    targets = np.stack(targets)

    if no_time_information:
        timesteps = [np.ones_like(e["t"][:-1]) for e in inputs]
    else:
        timesteps = [np.diff(e["t"]) for e in inputs]

    if len(resolution) == 1:
        tokens = [_tokenize_1d(e) for e in inputs]
    elif len(resolution) == 2:
        tokens = [_tokenize_2d(e, resolution[0]) for e in inputs]
    else:
        raise ValueError(f"resolution must have 1 or 2 dims, got {resolution}")

    lengths = np.array([len(t) for t in timesteps], dtype=np.int32)
    pad_length = (lengths.max() // pad_unit + 1) * pad_unit

    tokens = np.stack([
        np.pad(t, (0, pad_length - len(t)), mode="constant", constant_values=-1)
        for t in tokens
    ])
    timesteps = np.stack([
        np.pad(t, (0, pad_length - len(t)), mode="constant", constant_values=0)
        for t in timesteps
    ])

    # tonic timestamps are in microseconds; the model expects milliseconds.
    timesteps = timesteps.astype(np.float32) / 1000.0

    if lengths.shape[0] == 1:
        lengths = lengths[None, ...]

    return tokens, targets, timesteps, lengths


# ---------------------------------------------------------------------------
# Variants for LRA / irregular / language tasks
# ---------------------------------------------------------------------------

def lra_text_collate_fn(batch, *, resolution, pad_unit, cut_mix=0.0,
                        no_time_information=False, tokenize="unique"):
    """Collate for LRA text-like datasets (ListOps, IMDB): flat 1D int tokens.

    Unlike the event-stream collate this keeps the full sequence length (no
    ``[:-1]`` slice) because the underlying S5 loader already produces dense
    token arrays rather than event streams with one-past-the-end boundaries.
    """
    inputs, targets = zip(*batch)
    targets = np.stack(targets)
    timesteps = [np.ones_like(e["t"], dtype=np.float32) for e in inputs]
    tokens = [e["x"].astype(np.int32) for e in inputs]

    lengths = np.array([len(t) for t in timesteps], dtype=np.int32)
    pad_length = (lengths.max() // pad_unit + 1) * pad_unit
    tokens = np.stack([
        np.pad(t, (0, pad_length - len(t)), mode="constant", constant_values=-1)
        for t in tokens
    ])
    timesteps = np.stack([
        np.pad(t, (0, pad_length - len(t)), mode="constant", constant_values=0)
        for t in timesteps
    ]).astype(np.float32)

    if lengths.shape[0] == 1:
        lengths = lengths[None, ...]
    return tokens, targets, timesteps, lengths


def lra_image_collate_fn(batch, *, resolution, pad_unit, cut_mix=0.0,
                         no_time_information=False, tokenize="unique"):
    """Collate for LRA Image / Pathfinder / Path-X. Sequences are fixed-length
    so no padding is needed."""
    inputs, targets = zip(*batch)
    targets = np.stack(targets)
    timesteps = np.stack([np.ones_like(e["t"], dtype=np.float32) for e in inputs])
    tokens = np.stack([e["x"].astype(np.int32) for e in inputs])
    lengths = np.array([timesteps.shape[-1]] * len(inputs), dtype=np.int32)
    if lengths.shape[0] == 1:
        lengths = lengths[None, ...]
    return tokens, targets, timesteps, lengths


def retrieval_collate_fn(batch, *, resolution, pad_unit, cut_mix=0.0,
                         no_time_information=True, tokenize="unique"):
    """Collate for LRA Retrieval (AAN): each item is a (doc_a, doc_b) pair.

    Stacks the two docs along the batch dimension so the first half of the
    batch is ``doc_a`` and the second half is ``doc_b``; the retrieval head
    splits them back and computes pairwise features.
    """
    pairs, targets = zip(*batch)
    a, b = zip(*pairs)
    targets = np.stack(targets)

    if no_time_information:
        ts_a = [np.ones_like(e["t"][:-1], dtype=np.float32) for e in a]
        ts_b = [np.ones_like(e["t"][:-1], dtype=np.float32) for e in b]
    else:
        ts_a = [np.diff(e["t"]).astype(np.float32) for e in a]
        ts_b = [np.diff(e["t"]).astype(np.float32) for e in b]

    tok_a = [e["x"][:-1].astype(np.int32) for e in a]
    tok_b = [e["x"][:-1].astype(np.int32) for e in b]

    len_a = np.array([len(t) for t in ts_a], dtype=np.int32)
    len_b = np.array([len(t) for t in ts_b], dtype=np.int32)
    pad_length = max(
        (len_a.max() // pad_unit + 1) * pad_unit,
        (len_b.max() // pad_unit + 1) * pad_unit,
    )

    def _pad(arrs, val):
        return np.stack([np.pad(a, (0, pad_length - len(a)), mode="constant",
                                constant_values=val) for a in arrs])

    tokens = np.concatenate([_pad(tok_a, -1), _pad(tok_b, -1)], axis=0)
    timesteps = np.concatenate([_pad(ts_a, 0), _pad(ts_b, 0)], axis=0).astype(np.float32)
    lengths = np.concatenate([len_a, len_b], axis=0)
    if lengths.shape[0] == 1:
        lengths = lengths[None, ...]
    return tokens, targets, timesteps, lengths


def irregular_collate_fn(batch, *, resolution, pad_unit, cut_mix=0.0,
                         no_time_information=False, tokenize="unique"):
    """Collate for PersonActivity / Walker — already fixed-length per item."""
    inputs, targets = zip(*batch)
    targets = np.stack(targets) if targets[0] is not None else None
    timesteps = np.stack([e["t"] for e in inputs]).astype(np.float32)
    tokens = np.stack([e["x"] for e in inputs])
    lengths = np.array([timesteps.shape[-1]] * len(inputs), dtype=np.int32)
    if lengths.shape[0] == 1:
        lengths = lengths[None, ...]
    return tokens, targets, timesteps, lengths


def eigenworms_collate_fn(batch, *, resolution, pad_unit, cut_mix=0.0,
                          no_time_information=False, tokenize="unique"):
    """EigenWorms collate — inputs are (L, 6) vector-valued observations."""
    inputs, targets = zip(*batch)
    targets = np.stack(targets)
    timesteps = np.stack([np.ones_like(e["t"], dtype=np.float32) for e in inputs])
    tokens = np.stack([e["x"] for e in inputs])
    lengths = np.array([timesteps.shape[-1]] * len(inputs), dtype=np.int32)
    if lengths.shape[0] == 1:
        lengths = lengths[None, ...]
    return tokens, targets, timesteps, lengths


def language_collate_fn(batch, *, resolution, pad_unit, cut_mix=0.0,
                        no_time_information=False, tokenize="unique"):
    """Collate for PTB / WikiText: seq2seq next-token prediction.

    Both ``tokens`` and ``targets`` are sequences of the same shape, padded to
    the same pad length so the downstream loss can apply a mask via ``lengths``.
    """
    inputs, targets = zip(*batch)
    tokens = [e["x"].astype(np.int32) for e in inputs]
    target_seqs = [e["x"].astype(np.int32) for e in targets]
    timesteps = [np.ones_like(e["t"], dtype=np.float32) for e in inputs]

    token_len = np.array([len(t) for t in tokens], dtype=np.int32)
    target_len = np.array([len(t) for t in target_seqs], dtype=np.int32)
    pad_length = max(
        (token_len.max() // pad_unit + 1) * pad_unit,
        (target_len.max() // pad_unit + 1) * pad_unit,
    )

    tokens = np.stack([np.pad(t, (0, pad_length - len(t)), mode="constant",
                              constant_values=-1) for t in tokens])
    target_seqs = np.stack([np.pad(t, (0, pad_length - len(t)), mode="constant",
                                   constant_values=-1) for t in target_seqs])
    timesteps = np.stack([np.pad(t, (0, pad_length - len(t)), mode="constant",
                                 constant_values=0) for t in timesteps]).astype(np.float32)

    if token_len.shape[0] == 1:
        token_len = token_len[None, ...]
    return tokens, target_seqs, timesteps, token_len
