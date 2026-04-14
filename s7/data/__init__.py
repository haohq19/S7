"""Dataset registry for S7.

Each dataset module exposes a ``build_<name>_loaders(cache_dir=..., **cfg)``
function that returns ``(train_loader, val_loader, test_loader, info)`` where
``info`` is a :class:`DatasetInfo` dataclass carrying ``n_classes``,
``num_embeddings``, and ``train_size``.

The registry maps Hydra task names (as in ``configs/task/<name>.yaml``) to
builder callables. All 15 tasks covered by the paper are natively ported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class DatasetInfo:
    n_classes: int
    num_embeddings: int
    train_size: int


# ---------------------------------------------------------------------------
# Native builder dispatch — lazy imports keep ``s7.data`` cheap to import
# even for users that only need a subset of the datasets.
# ---------------------------------------------------------------------------

def _dvs_gesture_builder(**kw):
    from s7.data.dvs_gesture import build_loaders
    return build_loaders(**kw)


def _shd_builder(**kw):
    from s7.data.spiking_audio import build_shd_loaders
    return build_shd_loaders(**kw)


def _ssc_builder(**kw):
    from s7.data.spiking_audio import build_ssc_loaders
    return build_ssc_loaders(**kw)


def _listops_builder(**kw):
    from s7.data.lra import build_listops_loaders
    return build_listops_loaders(**kw)


def _text_builder(**kw):
    from s7.data.lra import build_text_loaders
    return build_text_loaders(**kw)


def _retrieval_builder(**kw):
    from s7.data.lra import build_retrieval_loaders
    return build_retrieval_loaders(**kw)


def _image_builder(**kw):
    from s7.data.lra import build_image_loaders
    return build_image_loaders(**kw)


def _pathfinder_builder(**kw):
    from s7.data.lra import build_pathfinder_loaders
    return build_pathfinder_loaders(**kw)


def _pathx_builder(**kw):
    from s7.data.lra import build_pathx_loaders
    return build_pathx_loaders(**kw)


def _eigenworms_builder(**kw):
    from s7.data.irregular import build_eigenworms_loaders
    return build_eigenworms_loaders(**kw)


def _personactivity_builder(**kw):
    from s7.data.irregular import build_personactivity_loaders
    return build_personactivity_loaders(**kw)


def _walker_builder(**kw):
    from s7.data.irregular import build_walker_loaders
    return build_walker_loaders(**kw)


def _ptb_builder(**kw):
    from s7.data.language import build_ptb_loaders
    return build_ptb_loaders(**kw)


def _wikitext2_builder(**kw):
    from s7.data.language import build_wikitext2_loaders
    return build_wikitext2_loaders(**kw)


def _wikitext103_builder(**kw):
    from s7.data.language import build_wikitext103_loaders
    return build_wikitext103_loaders(**kw)


DATASETS: Dict[str, Callable] = {
    "dvs-gesture-classification": _dvs_gesture_builder,
    "shd-classification": _shd_builder,
    "ssc-classification": _ssc_builder,
    "listops-classification": _listops_builder,
    "text-classification": _text_builder,
    "retrieval-classification": _retrieval_builder,
    "image-classification": _image_builder,
    "pathfinder-classification": _pathfinder_builder,
    "pathx-classification": _pathx_builder,
    "eigenworms-classification": _eigenworms_builder,
    "personactivity-classification": _personactivity_builder,
    "walker-classification": _walker_builder,
    "ptb-classification": _ptb_builder,
    "wikitext2-classification": _wikitext2_builder,
    "wikitext103-classification": _wikitext103_builder,
}


def get_builder(task_name: str) -> Callable:
    """Look up a dataset builder, raising a helpful error on typos."""
    if task_name not in DATASETS:
        raise KeyError(
            f"unknown task '{task_name}'. Registered: {sorted(DATASETS)}"
        )
    return DATASETS[task_name]


__all__ = ["DatasetInfo", "DATASETS", "get_builder"]
