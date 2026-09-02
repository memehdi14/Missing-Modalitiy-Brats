# Brain Tumor Segmentation Under Missing Modalities — 21-Day Implementation Plan

**Goal reframed:** publication is OFF the table for this cycle. This is a learning project.
**What you're actually building, in priority order:**
1. A working missing-modality-robust segmentation pipeline on BraTS (baseline + from-scratch transformer)
2. A custom fused attention kernel (Triton), written and benchmarked by you
3. A GUI to demo the whole thing (upload/select a case, toggle modalities on/off, see the segmentation + kernel speed comparison live)

No SOTA-chasing, no novelty pressure, no ablation-for-reviewers busywork. Every step exists because it teaches you something concrete: data engineering, from-scratch transformers, GPU kernel programming, and shipping a demoable tool.

**GPU:** confirmed — 16GB laptop (A5000 laptop variant). Patch size default: **96³**, with mixed precision (`torch.cuda.amp`) non-optional from Day 5 onward. Batch size will likely be forced to 1-2 once the transformer is attached (more memory-hungry than the CNN baseline) — use gradient accumulation (4-8 steps) to keep the effective batch size reasonable rather than fighting for a larger real batch size. Try 128³ only for the baseline U-Net alone (no transformer) if 96³ trains comfortably with room to spare; drop back to 96³ the moment the transformer is wired in.

---

## Architecture (what you're implementing, from scratch)

You are implementing a **masked-fusion transformer** — this exact idea (attention masking over mmFormer-style placeholder tokens) is established in the literature (M2FTrans and others already did this as a published contribution), which is fine and actually helpful for you: it means there's a well-defined target to implement against, with known behavior to sanity-check your implementation. You're not inventing the idea, you're building it yourself to understand it.

```
For each of 4 modalities (if present):
   Modality-specific 3D CNN stem (3-4 conv stages, shared architecture, separate weights)
        -> feature map per modality -> flatten into patch tokens + learnable 3D positional embeddings

Concatenate tokens from all PRESENT modalities into one sequence
        -> absent modality's tokens never enter the sequence at all (this is "masking")

Your from-scratch Transformer Encoder:
   - Scaled dot-product attention, multi-head (built with nn.Linear/matmul, NOT nn.MultiheadAttention)
   - Attention mask applied so tokens only attend to present-modality tokens
   - LayerNorm, residual connections, feed-forward block
   - Stack of N blocks

CNN Decoder (U-Net style):
   - Upsampling stages with skip connections back to encoder feature maps
   - Final 1x1x1 conv -> 3-channel sigmoid output (WT, TC, ET)
```

**Baseline (build first, always):** 3D U-Net + random modality dropout (zero-fill, no masking) — this is what you compare your transformer against, and what tells you the transformer is actually doing something.

---

## Attention foundations — do this before Day 7, not on Day 7

You know CNNs solidly but attention/transformers are new territory. Don't let Day 7 be the first time you meet these ideas — copying code you don't understand defeats the point of "from scratch." Spend a real, focused session (evening between Day 5-6 is a good slot) building the intuition first:

1. **"The Illustrated Transformer"** by Jay Alammar — visual, no heavy math, builds the intuition for what Q/K/V are and why attention works at all. Read this first.
2. **"Attention Is All You Need"** (Vaswani et al.) — Section 3 only, now that you have intuition from step 1.
3. **"The Annotated Transformer"** (Harvard NLP) — full from-scratch PyTorch walkthrough, line by line. This is your implementation reference for Day 7-10. Read it AND run it.

**Self-check before moving to Day 7:** can you explain, in your own words, what Q, K, and V represent, and why "attention" is really just a weighted average where the weights are learned from how well things match each other? If not yet — that's fine, just don't skip ahead until it clicks. This is the one place in the whole plan where understanding matters more than speed.

---

## The kernel

**What:** a fused Triton kernel for masked multi-head self-attention — QKᵀ → apply modality-presence mask → softmax → weighted-V in one kernel, instead of PyTorch's multi-step version (which materializes a mask tensor and does separate softmax/matmul ops with extra memory traffic).

**Why this one:** it's small enough to finish in ~5 days, it plugs directly into the architecture you're already building (not a bolt-on), and "does my kernel match PyTorch's output, and is it faster" is a clean, satisfying thing to verify and demo.

**Steps (do not skip the order):**
1. Get the model training correctly with plain PyTorch masked attention first. Correctness before speed, always.
2. Write the Triton kernel to compute the same operation.
3. Unit test: `torch.allclose(triton_output, pytorch_output, atol=1e-3)` across several input shapes, including edge cases (1 modality present, all 4 present).
4. Benchmark: wall-clock time + peak memory, PyTorch vs. Triton, across a few sequence lengths/batch sizes. Plot this — it's your best demo visual.
5. Swap into the real model, retrain briefly, confirm no accuracy regression.

---

## The GUI (for presenting the work)

Use **Streamlit** or **Gradio** — either is fast to build in 1-2 days, no need to hand-roll a frontend.

**What it should show:**
- A case selector (pick a BraTS validation case)
- Modality toggles (checkbox for T1 / T1ce / T2 / FLAIR) — turning modalities on/off and re-running inference live is the single best way to make "missing-modality robustness" tangible to someone watching
- Side-by-side: input modality slices (whatever's toggled on) + ground truth mask + your model's predicted mask, overlaid on a brain slice
- A slider to scrub through slices (z-axis) of the 3D volume
- A small panel showing the kernel benchmark result (naive PyTorch vs. Triton — bar chart of time/memory)
- Optional if time allows: Dice score displayed live for the currently selected modality combination

**Keep it simple.** This does not need to be production software — it needs to make your two contributions (missing-modality robustness, kernel speed) visible and interactive in under 2 days of build time. Don't scope-creep it into more than that.

---

## Code structure

```
project/
├── data/
│   ├── dataset.py             # BraTS loader, patch sampling, modality dropout
│   └── preprocess.py          # z-score norm, brain crop
├── models/
│   ├── unet_baseline.py       # 3D U-Net + random dropout baseline
│   ├── attention.py           # from-scratch scaled dot-product + multi-head attention
│   ├── transformer_block.py   # encoder block: attention + FFN + norm + residual
│   ├── stem.py                # per-modality CNN stem + positional embedding
│   ├── decoder.py              # U-Net style decoder
│   └── full_model.py           # wires stem + transformer + decoder together
├── kernels/
│   ├── naive_masked_attention.py    # PyTorch reference
│   ├── triton_masked_attention.py   # your Triton kernel
│   ├── test_kernel_correctness.py   # allclose unit tests
│   └── bench_kernel.py              # speed/memory benchmark
├── toy_task/
│   └── copy_task.py            # validate your from-scratch attention on a trivial task first
├── train.py                    # training loop (works for both baseline and full model)
├── eval.py                     # Dice/HD95 across modality combinations
├── gui/
│   └── app.py                  # Streamlit/Gradio demo
└── configs/                    # one config per run (baseline, transformer, kernel-swapped)
```

---

## 21-Day Plan

| Day | Task |
|---|---|
| **1** | *(done)* Problem understanding, literature landscape, honest novelty check |
| **2** | Env setup, confirm GPU/VRAM, download BraTS2021, inspect one case, confirm label taxonomy (0/1/2/4) |
| **3** | Preprocessing: z-score norm (brain-mask voxels only), patch extraction (start 128³, fallback 96³), tumor-biased sampling |
| **4** | `Dataset` class complete, train/val/test split saved to fixed files, modality dropout logic (uniform random, all non-empty subsets reachable) |
| **5** | 3D U-Net baseline implementation (instance/group norm, sigmoid multi-label output), Dice+BCE loss, training loop skeleton |
| **6** | Eval harness: Dice/HD95 computation, sliding-window inference for full volumes, loop over modality combinations (start with 5 representative, all 15 later if time allows) |
| **6.5** | *(evening, between Day 5-6)* Attention foundations — Illustrated Transformer → Attention Is All You Need §3 → Annotated Transformer. Don't skip this if attention is new to you. |
| **7** | Kick off baseline training run (let it run in background); toy-task attention validation starts here in parallel — scaled dot-product + multi-head attention from scratch, test on a toy copy/sort task |
| **8** | Baseline training likely finishing — check results table, sanity vs. published Dice ranges. Finish toy-task attention validation, confirm it actually learns. |
| **9** | 3D positional embeddings, per-modality CNN stem, patch tokenization — get shapes flowing correctly with a dummy forward pass |
| **10** | Wire full transformer encoder block (attention + FFN + norm + residual), stack N blocks, confirm forward pass works end-to-end on dummy input |
| **11** | Wire stem → transformer → decoder into `full_model.py`, get it training on full-modality input (no missing modalities yet) |
| **12** | Debug convergence — confirm your model reaches roughly baseline-level Dice on full-modality input before adding missing-modality complexity |
| **13** | Add attention masking for missing modalities + modality dropout training |
| **14** | Extend eval to run your model across modality combinations, compare vs. baseline U-Net numbers |
| **15** | Finalize naive PyTorch masked-attention reference implementation used by your model; write kernel correctness test harness |
| **16** | Write the Triton kernel — first pass, get it running (may not be numerically correct yet) |
| **17** | Debug Triton kernel until `allclose` passes against the naive PyTorch version across shapes/edge cases |
| **18** | Benchmark kernel: speed + memory, PyTorch vs. Triton, across a few sizes. Make the plot. |
| **19** | Swap kernel into `full_model.py`, retrain briefly, confirm no Dice regression |
| **20** | Build the GUI: case selector, modality toggles, slice viewer, prediction overlay, kernel benchmark panel |
| **21** | Polish GUI, dry-run the full demo end-to-end, buffer for whatever slipped |

**If you fall behind, cut in this order:** GUI polish (keep it minimal, functional) → full 15-combo eval (5 combos is enough for a learning project) → kernel benchmark plot variety (one clean comparison is enough). **Never cut:** the baseline, the from-scratch attention toy-task validation, and the kernel correctness test — these three are where the actual learning happens.

---

## Checklist

- [ ] GPU/VRAM confirmed, patch size decided accordingly (16GB laptop → 96³ default)
- [ ] Attention foundations understood — can explain Q/K/V in your own words before starting Day 7
- [ ] BraTS2021 downloaded, inspected, label taxonomy confirmed (0/1/2/4, no "3")
- [ ] Preprocessing pipeline working, patches sanity-checked visually
- [ ] Baseline 3D U-Net + dropout trained, results table produced
- [ ] From-scratch attention validated on a toy task (actually learns, not just runs)
- [ ] Full transformer model built, trains on full-modality input to a reasonable Dice
- [ ] Attention masking + modality dropout added, missing-modality eval working
- [ ] Naive PyTorch masked attention correct and used in training
- [ ] Triton kernel implemented, passes correctness tests against naive version
- [ ] Kernel benchmarked (speed + memory), plot made
- [ ] Kernel swapped into full model, no Dice regression confirmed
- [ ] GUI built: modality toggles, slice viewer, prediction overlay, kernel benchmark panel
- [ ] Full demo dry-run completed end-to-end
