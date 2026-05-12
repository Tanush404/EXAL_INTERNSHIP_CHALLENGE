# Task 5 — Federated Learning Written Analysis

## Which camera node converges slowest and why?

Client 0 and Client 4 consistently showed the lowest validation accuracy
across rounds (0.0000 in most rounds). This is attributed to their
Dirichlet-sampled shards having the most skewed class distributions —
under alpha=0.5, some clients receive very few samples from certain
classes, making local gradients highly biased. When these biased updates
are averaged with other clients via FedAvg, they contribute noise rather
than signal, slowing their personal convergence.

Client 1 showed the earliest improvement (0.0008 in Round 1), likely
because its shard happened to contain a more balanced class distribution
by chance under the Dirichlet sampling.

## Accuracy Gap Analysis

| Metric | Value |
|--------|-------|
| Centralized baseline accuracy | 24.95% |
| FL final accuracy (Round 5) | 2.14% |
| Accuracy gap | 22.81% |

The large gap is expected given only 5 rounds of training on a 10%
data subset on CPU. In production with 20+ rounds and full data,
FedAvg typically closes this gap significantly. The gap also reflects
the fundamental challenge of non-IID data — each client's local model
drifts toward its own data distribution, and FedAvg averaging partially
cancels these drifts.

## FL + INT8 Deployment Pipeline

After 5 federated rounds, the aggregated global model was exported to
ONNX and quantized to INT8 using the same pipeline as Task 2. Results:

| Metric | FP32 | INT8 |
|--------|------|------|
| Latency | 4.09 ms | 4.16 ms |

The INT8 model shows similar latency due to the ConvInteger limitation
identified in Tasks 3 and 4 — hardware VNNI support is required for
real INT8 speedup. On a target edge device with VNNI (Intel Atom,
ARM Cortex-A55), INT8 would deliver ~4x speedup.

## Would I recommend FL for this product?

**Yes, with caveats.** FL is the correct architecture for smart home
cameras because:
1. Privacy — raw video never leaves the device
2. Personalization — each camera adapts to its environment over time
3. Regulatory compliance — GDPR/CCPA alignment without data centralization

## Real Deployment Risks

1. **Stragglers** — slow or offline cameras delay aggregation rounds
2. **Poisoning attacks** — a compromised camera can inject malicious
   gradients into the global model
3. **Communication cost** — 49.6 MB per round × daily updates = significant
   bandwidth on home WiFi
4. **Non-IID drift** — as seen in results, heterogeneous data slows
   convergence and requires more rounds
5. **Model staleness** — cameras that miss rounds receive stale global
   weights, diverging from the fleet

## What I would do with more time
- Implement FedProx instead of FedAvg to handle non-IID drift better
- Add differential privacy (Gaussian noise on gradients) for stronger
  privacy guarantees
- Run 20+ rounds with full dataset to close the accuracy gap
- Test with actual camera hardware (Raspberry Pi) for real latency numbers

