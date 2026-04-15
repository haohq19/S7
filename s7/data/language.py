"""Language modeling datasets: Penn Treebank, WikiText-2, WikiText-103.

All three share the same word-level tokenization pipeline (``Dictionary`` +
``Corpus``) and differ only in vocabulary size and default sequence length.
The raw text files are expected at ``<cache_dir_parent>/{ptb,wikitext2,wikitext103}/
{train,valid,test}.txt`` (the standard Mikolov layout).
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tonic.io import make_structured_array
from torch.utils.data import DataLoader, Dataset

from s7.data import DatasetInfo
from s7.data.collate import language_collate_fn


# ---------------------------------------------------------------------------
# Word-level dictionary + corpus (Mikolov-style)
# ---------------------------------------------------------------------------

class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return token_id

    def __len__(self) -> int:
        return len(self.idx2word)


class Corpus:
    """Tokenize a ``{train,valid,test}.txt`` trio into ``torch.LongTensor`` ids.

    On first access the token tensors are cached to ``<path>/corpus_cache.pkl``
    so subsequent runs skip the scan.
    """

    def __init__(self, path: os.PathLike, cache_dir: Optional[os.PathLike] = None):
        self.dictionary = Dictionary()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir and (self.cache_dir / "corpus_cache.pkl").exists():
            with open(self.cache_dir / "corpus_cache.pkl", "rb") as f:
                cached = pickle.load(f)
            self.dictionary = cached["dictionary"]
            self.train = cached["train"]
            self.valid = cached["valid"]
            self.test = cached["test"]
            return

        self.train = self._tokenize(os.path.join(path, "train.txt"))
        self.valid = self._tokenize(os.path.join(path, "valid.txt"))
        self.test = self._tokenize(os.path.join(path, "test.txt"))

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_dir / "corpus_cache.pkl", "wb") as f:
                pickle.dump(
                    {
                        "dictionary": self.dictionary,
                        "train": self.train,
                        "valid": self.valid,
                        "test": self.test,
                    },
                    f,
                )

    def _tokenize(self, path: str) -> torch.LongTensor:
        assert os.path.exists(path), path
        # First pass: build vocab.
        total = 0
        with open(path, "r") as f:
            for line in f:
                words = line.split() + ["<eos>"]
                total += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # Second pass: emit ids.
        ids = torch.LongTensor(total)
        i = 0
        with open(path, "r") as f:
            for line in f:
                for word in line.split() + ["<eos>"]:
                    ids[i] = self.dictionary.word2idx[word]
                    i += 1
        return ids


# ---------------------------------------------------------------------------
# Sliding-window seq2seq dataset
# ---------------------------------------------------------------------------

class _LMDataset(Dataset):
    """Emits ``(input_ids[i:i+L], target=input_ids[i+1:i+1+L])`` pairs.

    ``L`` is drawn from a jittered distribution around ``seq_len`` — a trick from
    SalesForce's AWD-LSTM training code that the original S7 repo also uses.
    """

    _DTYPE = np.dtype([("t", int), ("x", int), ("p", int)])

    def __init__(self, corpus_split: torch.LongTensor, seq_len: int):
        self.data = corpus_split
        self.seq_len = seq_len

    def __len__(self):
        return (self.data.size(0) - 1) // self.seq_len

    def __getitem__(self, idx):
        base = idx * self.seq_len
        bptt = self.seq_len if np.random.random() < 0.95 else self.seq_len / 2
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        seq_len = min(seq_len, len(self.data) - 1 - base)
        input_ids = self.data[base : base + seq_len]
        target_ids = self.data[base + 1 : base + 1 + seq_len]
        times = np.arange(len(input_ids))
        events = make_structured_array(times, input_ids.numpy(), 1, dtype=self._DTYPE)
        target = make_structured_array(times, target_ids.numpy(), 1, dtype=self._DTYPE)
        return events, target


def _lm_loaders(
    *,
    data_subdir: str,
    vocab_size: int,
    seq_len: int,
    cache_dir: str,
    per_device_batch_size: int,
    per_device_eval_batch_size: int,
    world_size: int,
    num_workers: int,
    seed: int,
    pad_unit: int,
    no_time_information: bool,
):
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    root = Path(cache_dir) / data_subdir
    corpus = Corpus(root, root)

    train = _LMDataset(corpus.train, seq_len)
    val = _LMDataset(corpus.valid, seq_len)
    test = _LMDataset(corpus.test, seq_len)

    collate = partial(
        language_collate_fn, resolution=(vocab_size,), pad_unit=pad_unit,
        no_time_information=no_time_information,
    )

    def _loader(ds, bsz, shuffle, drop_last):
        return DataLoader(
            ds,
            batch_size=bsz,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate,
            num_workers=num_workers,
            generator=g,
            persistent_workers=num_workers > 0,
        )

    train_loader = _loader(train, per_device_batch_size * world_size, True, True)
    val_loader = _loader(val, per_device_eval_batch_size * world_size, False, False)
    test_loader = _loader(test, per_device_eval_batch_size * world_size, False, False)
    info = DatasetInfo(
        n_classes=vocab_size, num_embeddings=vocab_size, train_size=len(train)
    )
    return train_loader, val_loader, test_loader, info


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_ptb_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 32,
    no_time_information: bool = False,
    seq_len: int = 70,
    **_unused,
):
    # PTB has 10K vocab
    return _lm_loaders(
        data_subdir="ptb", vocab_size=10000, seq_len=seq_len,
        cache_dir=cache_dir,
        per_device_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        world_size=world_size, num_workers=num_workers,
        seed=seed, pad_unit=pad_unit,
        no_time_information=no_time_information,
    )


def build_wikitext2_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 32,
    no_time_information: bool = False,
    seq_len: int = 70,
    **_unused,
):
    # WikiText-2 vocabulary is ~84.6K
    return _lm_loaders(
        data_subdir="wikitext2", vocab_size=84608, seq_len=seq_len,
        cache_dir=cache_dir,
        per_device_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        world_size=world_size, num_workers=num_workers,
        seed=seed, pad_unit=pad_unit,
        no_time_information=no_time_information,
    )


def build_wikitext103_loaders(
    *,
    cache_dir: str,
    per_device_batch_size: int = 32,
    per_device_eval_batch_size: int = 64,
    world_size: int = 1,
    num_workers: int = 0,
    seed: int = 42,
    pad_unit: int = 32,
    no_time_information: bool = False,
    seq_len: int = 140,
    **_unused,
):
    # WikiText-103 vocabulary is ~267.7K
    return _lm_loaders(
        data_subdir="wikitext103", vocab_size=267735, seq_len=seq_len,
        cache_dir=cache_dir,
        per_device_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        world_size=world_size, num_workers=num_workers,
        seed=seed, pad_unit=pad_unit,
        no_time_information=no_time_information,
    )
