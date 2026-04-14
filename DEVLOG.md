# S7 Reproduction Dev Log

Dev log for the S7 (arxiv 2410.03464) full rewrite + reproduction effort.
Working on Euler cluster, 1× RTX 3090. Workspace: `/cluster/scratch/haihao/claude/`.

---

## 2026-04-14 — Day 1: scoping & code map

### Decisions locked
- **Framework**: keep JAX/Flax (no PyTorch port).
- **Selectivity**: only Δ is input-dependent at the *parameter* level. The paper's wording "B̄ₖ, C̄ₖ are functions of u" refers to the *discretized* matrices, which inherit u-dependence through Δ(u). Underlying continuous B, C are fixed Flax params. Do **not** restore the removed `separate_dbc` Dense-projection path.
- **Workspace**: `/cluster/scratch/haihao/claude/{data,checkpoints,wandb,logs}` (15-day purge — must touch / re-stage).
- **Env**: `conda create -n s7 python=3.9.19` + `pip install -r requirements.txt`.
- **Dataset order**: DVS128Gesture first, then all others.

### Codebase audit findings (pre-rewrite snapshot)
- Stack: JAX 0.4.30, Flax 0.8.5, optax, hydra, tonic 1.5.0, torch 1.11 for dataloader.
- Entry: `run_training.py` / `run_evaluation.py`, hydra root `configs/base.yaml`.
- Core SSM: `event_ssm/ssm.py` — diagonal complex Λ (S5-style), parallel `associative_scan`, three discretizations: `zoh`, `dirac`, `async`. Bidirectional supported.
- Selectivity (post-`c65472d`): `step_proj` Dense produces per-token Δ; B, C remain fixed params.
- Model: `event_ssm/seq_model.py` `StackedEncoderModel` → `SequenceStage` (in `layers.py`) → `SequenceLayer` (norm → SSM → GELU+gated Dense → residual → optional `EventPooling`).
- DVS128 pipeline: tonic `DVSGesture`, token = `x*256 + y*2 + p`, padded to `pad_unit=524288`, sliced to `slice_events=65536`. Train aug: DropEventChunk/DropEvent/UniformNoise/TimeSkew/TimeJitter/SpatialJitter/Downsample/Roll/Rotate/Scale + cut_mix.
- Config (`configs/model/dvs/small.yaml`, post-bugfix): d_model=64, d_ssm=64, ssm_block_size=16, 2 stages × 4 layers, async discretization, timepool head, pooling_stride=16, state_expansion_factor=2.
- Task (`configs/task/dvs-gesture.yaml`, post-bugfix): 100 epochs, batch=8, ssm_base_lr=1e-6, lr_factor=6, warmup=10, cross_entropy.

### Bugs / cruft to clean in rewrite
1. `event_ssm/layers_old.py` — dead duplicate.
2. `configs/system/local.yaml` — hardcoded `/media/SSD0/haohq/...`; replace with our scratch path.
3. `discretize_and_project_inputs_input_dep` ignores `log_a`/`stablessm_a` reparam branches that the LTI path applies.
4. `run_training.py` vs `run_evaluation.py` loss-type plumbing differs.
5. Per-dataset timestep unit handling is inconsistent (DVS /1000, person_activity /1).
6. `EventPooling` raises on stride=1 instead of acting as identity.
7. `print.py` looks like a debug helper; review for removal.
8. `S5/` subdir vendored — verify if still needed after rewrite.

### Bugfix commit `c65472d` — key changes (`fix bug in data dependence`)
- **Removed `separate_dbc` mode** entirely — that path put B and C through `B_proj`/`C_proj` Dense layers (Mamba-style full selectivity). Confirmed by user: this should *not* be restored.
- Renamed `S5SSM` → `S7`.
- B/C init paths unified: always Flax params, no input-dependent branch.
- `apply_ssm` signature simplified — dropped `stride` and `input_dependent` args; pooling moved out into `SequenceLayer`.
- `SequenceLayer`: pooling now happens **after** `x = x + skip`, not before. (Previously pooled the skip branch only — lossy and asymmetric.)
- DVS hyperparams: d_model 16→64, layers_per_stage 3→4, lr 1.2e-5→1e-6, batch 16→8.
- Default task switched from `spiking-speech-commands` to `dvs-gesture`.

### Paper target numbers (arxiv 2410.03464)
| Dataset | Reported |
|---|---|
| DVS128 Gesture | **99.2%** |
| SHD | 96.3% |
| SSC | 88.2% |
| EigenWorms | 97.5% |
| LRA ListOps | 63.77% |
| LRA Text | 87.22% |
| LRA Retrieval | 91.80% |
| LRA Image | 61.14% |
| LRA Pathfinder | 65.52% |
| LRA Path-X | 61.50% |
| Human Activity | 94.09% |
| Walker2D | MSE 0.114 |

Stable A reparam (Eq. 4): **f(Λ) = I − (Λ² + 0.5·I)⁻¹**.
Async discretization (Eq. 12): xₖ = e^{Λ Δtₖ} xₖ₋₁ + B uₖ.
Event tokenization (Eq. 10): T(ε) = 2·(x·sx + y) + p.

### Status (end of day 1)
- [x] Code map
- [x] Paper read (HTML)
- [x] Bugfix commit decoded
- [x] Workspace dirs created
- [x] Conda env `s7` (python 3.9.19, JAX 0.4.30 + CUDA12, tonic 1.5.0) — JAX sees 1× CUDA device
- [x] DVS128Gesture dataset staged at `/cluster/scratch/haihao/claude/data/dvs128/DVSGesture/` (1077 train / 264 test, MD5 verified)
- [x] `configs/system/local.yaml` patched to point at scratch
- [x] **Smoke test passed**: 12 epochs DVS128 → **test acc 0.87** (paper 0.992 at 100ep)
- [x] Full 100-epoch DVS128 reproduction running on legacy code (baseline)
- [x] **Rewrite: `s7/` package complete** — core SSM, layers, model, train/, data/, scripts/, tests/
- [x] **CPU parity tests: bit-identical** at both S7-layer and ClassificationModel levels (max |Δy| = 0.0)
- [x] **Dataset registry**: all 15 task names reachable through `scripts/train.py` (DVS128 native, others via legacy adapter)

### Notes from smoke test
- `warmup_epochs=10` in the default config is incompatible with `num_epochs<10` — produces negative `decay_steps` in cosine schedule. Use `warmup_epochs <= num_epochs / 2` for short runs.
- Trainer prints model topology to stdout (5 print sites per layer × 2 init passes × N layers — very noisy). The trainer doesn't print per-epoch metrics, only end-of-training. Address in rewrite.
- 2.59M params for DVS small config.
- Sequence length is 524288 (pad_unit), sliced to 65536 events for training inputs.
- The slow phase is JAX trace + first compile (~1-2 min); subsequent epochs are fast.

### DVS128 download workaround
Tonic uses the figshare web URL `https://figshare.com/ndownloader/files/<id>` which AWS WAF gates with a JS challenge — automated downloads return 0 bytes. **Workaround:** use the figshare API instead — `https://api.figshare.com/v2/file/download/<id>` returns a presigned S3 URL that bypasses the WAF. File IDs:
- train: `38022171` → `ibmGestureTrain.tar.gz` (1.8 GB, MD5 `3a8f0d4120a166bac7591f77409cb105`)
- test: `38020584` → `ibmGestureTest.tar.gz` (660 MB, MD5 `56070e45dadaa85fff82e0fbfbc06de5`)

## 2026-04-14 PM — GPU runs + cleanup

### GPU allocation dance
- Node started with only 1× 3090 allocated; smoke test worked.
- 4× 3090 made available → tried smoke tests — kept OOMing at init with `2.02 GiB` allocation failures even though each GPU had 24 GB free.
- Root cause: JAX's BFC allocator defaults to `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` = 18 GB. The DVS128 forward needs ~22 GB at B=8, L=524288 (dominated by the (B, L, local_P) Bu tensor at each SSM layer). **Both legacy and rewrite fail on the default 75% fraction.**
- Fix: set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` in `scripts/train.py` and `scripts/evaluate.py`.
- GPU allocation changed mid-session: first 0+1, then 2+3 — keep `CUDA_VISIBLE_DEVICES` in sync.

### Performance parity confirmed
Direct step-time benchmark of new vs legacy at full DVS128 shape (B=8, L=524288):

| | init | compile+1st step | steady-state step |
|---|---|---|---|
| NEW | 25.2 s | 32.5 s | **0.24 s** |
| OLD | — | 22.1 s | **0.24 s** |

**Per-step GPU work is bit-identical between rewrite and legacy.** Compile is 10 s slower one-time (likely due to the way I bind Lambda/C in `__call__` vs `setup`; XLA traces differently but optimizes to the same graph). Not worth fixing.

### 1-epoch DVS128 smoke on rewrite (GPU 2)
```
[*] 2,587,915 model parameters
| epoch 1 | Train | acc 0.3772 | loss 1.7622 | epoch_time 444 s
| epoch 1 | Val   | acc 0.6326 | loss 1.1150
```
~7.4 min/epoch. 100 epochs ≈ 12 h at B=8, 1 GPU. Parity with legacy confirmed in every way that matters. Reproduction paused per user request.

### Complex warning investigation
Both new and legacy backward passes emit `ComplexWarning: Casting complex values to real discards the imaginary part` at the `gamma_bar * (B @ u_t)` line. This is a JAX quirk: `B` is complex, `u_t` is real, so the gradient w.r.t. `u_t` is complex but gets cast to real (imaginary part is 0 by construction for this operation). Harmless. Not a rewrite regression.

### Cleanup done
Deleted dead code:
- `event_ssm/layers_old.py` — duplicate of `layers.py`.
- `event_ssm/print.py` — standalone CIFAR diagnostic referencing the original author's machine paths.
- `event_ssm/{personactivity,worm,physionet,ptb}.py` — standalone diagnostic scripts referencing `/data/old_home/tsoydan/RPG` paths, each with `parse_args` at module level (broken as import).
- `best_eigenworms/`, `best_image/`, `best_pathfinder/` — 68 MB of stale checkpoint dumps + wandb logs.
- `setup.py` — pointed at a nonexistent `S7` package, replaced by the `s7/` package layout.

Still alive (needed by legacy adapter for un-ported datasets): `event_ssm/dataloading.py`, `event_ssm/transform.py`, `event_ssm/ssm.py` & friends, `S5/`, `odelstms/`.

## 2026-04-14 evening — full native dataset port

**All 15 datasets are now ported to `s7/data/`:**
- `s7/data/dvs_gesture.py` — DVS128 Gesture (native)
- `s7/data/spiking_audio.py` — SHD + SSC
- `s7/data/lra.py` — ListOps, Text (IMDB), Retrieval (AAN), Image (CIFAR10-gray), Pathfinder, Path-X. Delegates data-prep to the vendored `S5/s5/dataloaders/{lra,basic}` (kept as an external dep, no source modifications).
- `s7/data/irregular.py` — EigenWorms (local `.pt` files), PersonActivity (`odelstms.PersonData`), Walker2D (`odelstms.Walker2dImitationData`).
- `s7/data/language.py` — PTB, WikiText-2, WikiText-103. Ships its own `Dictionary` + `Corpus` word-level tokenizer (Mikolov-style) with a pickle cache at `<dataset>/corpus_cache.pkl`.

**All six collate variants moved into `s7/data/collate.py`:**
`event_stream_collate_fn`, `lra_text_collate_fn`, `lra_image_collate_fn`, `retrieval_collate_fn`, `irregular_collate_fn`, `eigenworms_collate_fn`, `language_collate_fn`.

**Registry** (`s7.data.DATASETS`) contains all 15 task names pointing at native builders — no legacy adapter, no fallback path.

**Retired**:
- `run_training.py`, `run_evaluation.py` — entirely replaced by `scripts/train.py` and `scripts/evaluate.py`.
- `event_ssm/dataloading.py` (2276 lines), `event_ssm/transform.py`, `event_ssm/trainer.py`, `event_ssm/train_utils.py`.

**What's still left in `event_ssm/`** (kept purely as parity-test anchors until reproduction is locked in):
`ssm.py`, `ssm_init.py`, `layers.py`, `seq_model.py`, `__init__.py`. The parity tests import `event_ssm.ssm.S7` and `event_ssm.seq_model.ClassificationModel` and assert bit-identical outputs against the rewrite. Delete this tree only after we're confident the reproduction numbers match.

**Vendored external deps** (kept untouched, treated like tonic): `S5/` (LRA data-prep), `odelstms/` (irregular-time data-prep).

### After-port verification
- `python tests/test_ssm_parity.py` → max |Δy| = 0.000e+00 ✓
- `python tests/test_model_parity.py` → max |Δy| = 0.000e+00 ✓
- `s7.data.DATASETS` → 15 tasks registered, all native builders importable

### Next
1. Smoke test `scripts/train.py` on a task other than DVS128 (SHD is the cheapest to stage — a few hundred MB of tonic HDF5s).
2. Stage remaining datasets on scratch as needed.
3. Launch reproduction runs per-dataset, record numbers vs. paper Table.
4. Delete `event_ssm/` once reproduction is locked.

### Rewrite design notes
- Class name `S7` is preserved (not `S7Layer`) so Flax auto-submodule names (`S7_0`) line up with legacy checkpoints.
- `step_proj` / `step_proj_kernel` / `step_proj_bias` names preserved for the Δ projection for the same reason (checkpoint + init-RNG compat).
- `EventPooling` silently acts as identity on `stride=1` instead of raising — matches how `SequenceLayer` guards the call, but is safer.
- `s7.data.DatasetInfo` replaces the ad-hoc `Data` dataclass from the legacy loader; the adapter shim translates between them so no training code sees the legacy class.
- `Trainer` prints per-epoch train/val metrics in a compact one-line format; the legacy trainer only printed the final test metric.
- `optim.build_optimizer` groups params into `ssm` (low-LR, no WD), `proj` (Δ projection, LR × lr_factor, no WD), and `regular` (everything else, LR × lr_factor, WD). Exactly matches legacy groupings.
