# Engineering Decision Log — Task 4
## Which variant would I ship and why?

### Summary of Results

| Model  | Threads | p50 Latency | p95 Latency | Accuracy |
|--------|---------|-------------|-------------|----------|
| FP32   | 1       | 7.77 ms     | 8.26 ms     | 0.20%    |
| FP32   | 2       | 5.04 ms     | 5.51 ms     | 0.20%    |
| FP32   | 4       | 3.57 ms     | 5.01 ms     | 0.20%    |
| INT8   | 1       | 55.40 ms    | 57.93 ms    | 0.20%    |
| INT8   | 2       | 46.79 ms    | 48.17 ms    | 0.20%    |
| INT8   | 4       | 48.34 ms    | 55.84 ms    | 0.20%    |
| Fused  | 1       | 7.83 ms     | 8.11 ms     | 0.20%    |
| Fused  | 2       | 4.95 ms     | 5.37 ms     | 0.20%    |
| Fused  | 4       | 5.09 ms     | 5.37 ms     | 0.20%    |

### Decision: Ship FP32 Fused with 2 Threads

I would ship the **operator-fused FP32 model with 2 threads**.

**Reasoning:**

1. **INT8 is slower on this hardware.** Counterintuitively, the INT8
   model is ~10x slower than FP32 (46-55ms vs 3-7ms). This is because
   the quantized model uses ConvInteger operators which require
   hardware-level INT8 SIMD support (VNNI instructions). The test CPU
   lacks these instructions, so ORT falls back to a slow software
   emulation path. On a target device with VNNI support (e.g. Intel
   Atom x6000, ARM Cortex-A55), INT8 would be the clear winner.

2. **Fused FP32 at 2 threads is the sweet spot.** At 4.95ms p50, it
   is faster than plain FP32 at 2 threads (5.04ms) due to eliminated
   redundant transposes and fused MatMul+Bias ops. Adding a 4th thread
   gives no benefit (5.09ms) — the overhead of thread synchronization
   cancels out the parallelism gain at this model size.

3. **Energy proxy.** Since RAPL is unavailable on WSL2, latency is
   used as an energy proxy (energy ≈ power × time). Lower latency
   at the same TDP means lower energy-per-inference. The Fused/2T
   variant minimises this proxy.

4. **Model size.** FP32 at 13.54MB fits comfortably in the RAM of
   most embedded Linux targets (Raspberry Pi 4: 4GB, Jetson Nano: 4GB).
   If the target were a microcontroller (<1MB RAM), INT8 with proper
   VNNI support would be mandatory.

### What I would do with more time
- Run on a device with VNNI support to validate INT8 speedup
- Measure actual RAPL energy on a native Linux machine
- Apply dynamic quantization as a middle ground between FP32 and
  static INT8 to avoid the ConvInteger bottleneck
- Profile with perf stat to identify memory-bound vs compute-bound ops

