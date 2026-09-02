# Hardware Specs & Training Notes

Reference this whenever you're setting batch size, patch size, or debugging OOM/thermal issues during training.

---

## Machine

- **GPU:** NVIDIA RTX A5000 **Laptop** GPU — 16GB VRAM (dedicated), not the 24GB desktop A5000. Don't assume desktop-A5000 numbers from any external benchmark/blog without checking if it's the laptop variant.
- **System RAM:** 127GB
- **OS:** Windows (confirmed via Task Manager) — decide WSL2 vs native Windows training env on Day 1, before writing code. WSL2 recommended: better CUDA/Linux tooling compatibility with most research repos.
- **Driver version:** 32.0.15.8092 (as of this check — confirm CUDA toolkit compatibility against this when installing PyTorch).
- **Idle GPU temp:** ~61°C at 0% utilization. This is a laptop thermal envelope — expect more aggressive throttling under sustained load than a desktop card. Monitor temp during long runs (Day 7 baseline training, and any Week 3+ transformer runs), especially multi-hour ones.

---

## What This Means for Training Config

| Setting | Plain U-Net baseline (Week 1) | Transformer fusion model (Week 3+) |
|---|---|---|
| Patch size | Try 128³ first | Likely need 96³ once attention is attached |
| Batch size | 1-2 realistic at 128³ | Likely forced to 1 |
| Mixed precision | Use from Day 4 onward — not optional at 16GB | Required |
| Gradient accumulation | Optional | Needed if batch size = 1, to keep effective batch size reasonable (accumulate 4-8 steps) |
| Norm layers | Instance/group norm (already planned — small batch sizes make batch norm unstable) | Same |

**Why mixed precision matters here specifically:** on a 16GB card, `torch.cuda.amp` (autocast + GradScaler) is often the difference between the model fitting in memory at all vs. not — treat it as a baseline requirement, not a later optimization pass.

**Attention memory scaling reminder:** the transformer's memory cost scales roughly with (token sequence length)². Your token sequence = sum of patch tokens across all *present* modalities, so a full-4-modality forward pass is the worst case for memory — test your patch size / tokenization stride against that worst case, not the missing-modality cases (those are cheaper by comparison).

---

## If You Hit OOM

Try in this order:
1. Turn on/confirm mixed precision is actually active.
2. Drop patch size (128³ → 96³).
3. Drop batch size to 1 + add gradient accumulation.
4. Increase tokenization stride / coarsen patch embedding (fewer, larger tokens — reduces sequence length quadratically for attention cost).
5. Only as a last resort: reduce transformer depth (fewer encoder blocks) or hidden dim — this changes the model, not just the memory footprint, so treat it as a design change worth noting if you do it.

---

## Thermal Monitoring

For any run expected to last more than ~1 hour (Day 7 baseline, all Week 3+ transformer runs): keep an eye on GPU temp during the first 15-20 minutes to see where it stabilizes under load. If you see throttling (unexpected drop in iterations/sec with temp near/above ~85-90°C sustained), that's a hardware constraint to note, not a training bug — don't waste time debugging code for a slowdown that's actually thermal.
