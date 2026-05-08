import onnxruntime as ort
import onnxoptimizer
import onnx
import numpy as np
import time
import matplotlib.pyplot as plt
import csv

# ── paths ──────────────────────────────────────────────────────────────────
FP32_MODEL  = "/home/durga/Task_3_CPP_Wrapper/mobilenet_v2.onnx"
INT8_MODEL  = "/home/durga/Task_3_CPP_Wrapper/mobilenet_v2_int8.onnx"
FUSED_MODEL = "/home/durga/Task_4_Power_Latency/mobilenet_v2_fused.onnx"

# ── dummy input (1×3×224×224 ImageNet-normalised) ──────────────────────────
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

# ── helper: measure latency for N warmup + M timed runs ───────────────────
def measure_latency(model_path, threads, n_warmup=10, n_runs=50):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # warmup
    for _ in range(n_warmup):
        sess.run(None, {input_name: dummy})

    # timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)   # ms

    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
    }

# ── Step 1: operator fusion on FP32 model ─────────────────────────────────
print("Applying operator fusion ...")
model = onnx.load(FP32_MODEL)
passes = [
    "fuse_consecutive_transposes",
    "eliminate_identity",
    "eliminate_nop_transpose",
    "fuse_transpose_into_gemm",
    "fuse_matmul_add_bias_into_gemm",
]
optimized = onnxoptimizer.optimize(model, passes)
onnx.save(optimized, FUSED_MODEL)
print(f"Fused model saved → {FUSED_MODEL}")

# ── Step 2: thread-count sweep ─────────────────────────────────────────────
models = {
    "FP32":  FP32_MODEL,
    "INT8":  INT8_MODEL,
    "Fused": FUSED_MODEL,
}
thread_counts = [1, 2, 4]
results = {}   # (model_name, threads) → latency dict

print("\n{'Model':<8} {'Threads':<8} {'p50 ms':<10} {'p95 ms':<10} {'p99 ms'}")
print("-" * 55)

for name, path in models.items():
    for t in thread_counts:
        try:
            lat = measure_latency(path, threads=t)
            results[(name, t)] = lat
            print(f"{name:<8} {t:<8} {lat['p50']:<10.2f} {lat['p95']:<10.2f} {lat['p99']:.2f}")
        except Exception as e:
            print(f"{name:<8} {t:<8} ERROR: {e}")
            results[(name, t)] = None

# ── Step 3: save CSV ───────────────────────────────────────────────────────
csv_path = "/home/durga/Task_4_Power_Latency/latency_results.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "threads", "p50_ms", "p95_ms", "p99_ms", "mean_ms"])
    for (name, t), lat in results.items():
        if lat:
            w.writerow([name, t, lat["p50"], lat["p95"], lat["p99"], lat["mean"]])
print(f"\nCSV saved → {csv_path}")

# ── Step 4: Pareto plot ────────────────────────────────────────────────────
# Accuracy values from Task 2
accuracy = {"FP32": 0.20, "INT8": 0.20, "Fused": 0.20}

fig, ax = plt.subplots(figsize=(8, 5))
colors = {"FP32": "steelblue", "INT8": "tomato", "Fused": "seagreen"}
markers = {1: "o", 2: "s", 4: "^"}

for (name, t), lat in results.items():
    if lat is None:
        continue
    ax.scatter(lat["p50"], accuracy[name],
               color=colors[name], marker=markers[t], s=120,
               label=f"{name} / {t}T", zorder=3)
    ax.annotate(f"{name}/{t}T",
                (lat["p50"], accuracy[name]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.set_xlabel("p50 Latency (ms)", fontsize=12)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
ax.set_title("Pareto Frontier: Accuracy vs Latency\n(WSL2 CPU — energy proxy = latency)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="lower right")
plt.tight_layout()

plot_path = "/home/durga/Task_4_Power_Latency/pareto_plot.png"
plt.savefig(plot_path, dpi=150)
print(f"Pareto plot saved → {plot_path}")
print("\nDone!")
