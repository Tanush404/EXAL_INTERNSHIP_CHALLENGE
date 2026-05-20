import numpy as np
import matplotlib.pyplot as plt
import csv
import os

np.random.seed(42)
N = 600  # 10 minutes at 1Hz

# ── Ground truth occupancy ──────────────────────────────────────────────────
ground_truth = np.zeros(N)
ground_truth[60:200]  = 1.0   # person in room 1min to 3.3min
ground_truth[300:450] = 1.0   # person returns 5min to 7.5min
ground_truth[500:560] = 1.0   # brief visit 8.3min to 9.3min

# ── Sensor simulation ───────────────────────────────────────────────────────
def simulate_pir(gt):
    """Binary PIR. 15% false trigger rate as specified."""
    readings = []
    for g in gt:
        if g == 1.0:
            readings.append(np.random.choice([1.0, 0.0], p=[0.85, 0.15]))
        else:
            readings.append(np.random.choice([0.0, 1.0], p=[0.85, 0.15]))
    return np.array(readings)

def simulate_temp(gt):
    """Temperature with 0.5C Gaussian drift when occupied."""
    readings = []
    for g in gt:
        if g == 1.0:
            readings.append(np.random.normal(1.0, 0.5))   # occupied: +1C drift
        else:
            readings.append(np.random.normal(0.0, 0.5))   # empty: no drift
    return np.array(readings)

def simulate_camera(gt):
    """Camera motion score 0.0-1.0 float, noisy."""
    readings = []
    for g in gt:
        if g == 1.0:
            readings.append(np.clip(np.random.normal(0.8, 0.15), 0, 1))
        else:
            readings.append(np.clip(np.random.normal(0.1, 0.1), 0, 1))
    return np.array(readings)

pir_readings    = simulate_pir(ground_truth)
temp_readings   = simulate_temp(ground_truth)
camera_readings = simulate_camera(ground_truth)

# Normalize temp to 0-1 range for fusion
temp_norm = np.clip((temp_readings - temp_readings.min()) /
                    (temp_readings.max() - temp_readings.min()), 0, 1)

# ── Save sensor CSV ─────────────────────────────────────────────────────────
csv_path = os.path.expanduser("~/Task_6_Sensor_Fusion/fusion_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "ground_truth", "pir", "temp_raw", "temp_norm", "camera"])
    for t in range(N):
        w.writerow([t, ground_truth[t], pir_readings[t],
                    temp_readings[t], temp_norm[t], camera_readings[t]])
print(f"CSV saved → {csv_path}")

# ── 2D Kalman Filter ─────────────────────────────────────────────────────────
class KalmanFilter2D:
    """
    2D Kalman Filter for occupancy estimation.

    State vector: [occupancy_score, rate_of_change]
    - occupancy_score: continuous confidence 0-1 that room is occupied
    - rate_of_change: how fast occupancy is changing (helps predict transitions)
    """
    def __init__(self, R_pir, R_temp, R_camera, Q_scale):
        # State: [occupancy_score, rate_of_change]
        self.x = np.array([[0.0], [0.0]])

        # State covariance — initial uncertainty
        self.P = np.eye(2) * 1.0

        # State transition matrix F
        # occupancy(t) = occupancy(t-1) + rate(t-1)*dt
        # rate(t) = rate(t-1)  (rate changes slowly)
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])

        # Process noise Q — how much can state change unexpectedly per step
        # Q_scale tunes trust in model vs sensors
        self.Q = np.array([[0.01, 0.0],
                           [0.0, 0.001]]) * Q_scale

        # Observation matrix H — sensors only observe occupancy_score, not rate
        self.H = np.array([[1.0, 0.0]])

        # Per-sensor measurement noise
        # R = how much we trust each sensor
        # Low R = trust sensor more, High R = trust model more
        self.R_pir    = np.array([[R_pir]])
        self.R_temp   = np.array([[R_temp]])
        self.R_camera = np.array([[R_camera]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Clip occupancy to valid range
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        return self.x[0, 0]

    def update(self, z, R):
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + R               # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y                          # update state
        self.P = (np.eye(2) - K @ self.H) @ self.P       # update covariance
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        return self.x[0, 0]

    def run(self, pir, temp, camera):
        estimates = []
        for t in range(len(pir)):
            self.predict()
            self.update(np.array([[pir[t]]]),    self.R_pir)
            self.update(np.array([[temp[t]]]),   self.R_temp)
            self.update(np.array([[camera[t]]]), self.R_camera)
            estimates.append(self.x[0, 0])
        return np.array(estimates)

# ── 3 Q/R configurations ─────────────────────────────────────────────────────
configs = {
    "Low R (trust sensors too much)": {
        "R_pir": 0.05, "R_temp": 0.05, "R_camera": 0.05,
        "Q_scale": 1.0,
        "desc": "R=0.05 — filter blindly follows every sensor spike"
    },
    "High R (trust model too much)": {
        "R_pir": 5.0, "R_temp": 5.0, "R_camera": 5.0,
        "Q_scale": 0.1,
        "desc": "R=5.0 — filter barely moves, ignores sensor changes"
    },
    "Balanced (optimal)": {
        "R_pir": 0.3, "R_temp": 0.5, "R_camera": 0.1,
        "Q_scale": 1.0,
        "desc": "R per-sensor — camera trusted most, temp least"
    },
}

results = {}
for name, cfg in configs.items():
    kf = KalmanFilter2D(cfg["R_pir"], cfg["R_temp"],
                        cfg["R_camera"], cfg["Q_scale"])
    estimates = kf.run(pir_readings, temp_norm, camera_readings)
    results[name] = estimates

# ── Metrics ──────────────────────────────────────────────────────────────────
threshold = 0.5
print("\n" + "="*55)
print("SENSOR FUSION RESULTS")
print("="*55)

for name, estimates in results.items():
    pred = (estimates > threshold).astype(int)
    tp = np.sum((pred == 1) & (ground_truth == 1))
    fp = np.sum((pred == 1) & (ground_truth == 0))
    tn = np.sum((pred == 0) & (ground_truth == 0))
    fn = np.sum((pred == 0) & (ground_truth == 1))
    acc = (tp + tn) / N
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.3f} | FAR: {far:.3f} | FRR: {frr:.3f}")

# PIR only baseline
pir_pred = (pir_readings > 0.5).astype(int)
pir_tp = np.sum((pir_pred == 1) & (ground_truth == 1))
pir_fp = np.sum((pir_pred == 1) & (ground_truth == 0))
pir_tn = np.sum((pir_pred == 0) & (ground_truth == 0))
pir_fn = np.sum((pir_pred == 0) & (ground_truth == 1))
pir_acc = (pir_tp + pir_tn) / N
pir_far = pir_fp / (pir_fp + pir_tn) if (pir_fp + pir_tn) > 0 else 0
pir_frr = pir_fn / (pir_fn + pir_tp) if (pir_fn + pir_tp) > 0 else 0
print(f"\nPIR only baseline:")
print(f"  Accuracy: {pir_acc:.3f} | FAR: {pir_far:.3f} | FRR: {pir_frr:.3f}")

# ── 3 Q/R tuning plots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
colors = {
    "Low R (trust sensors too much)": "tomato",
    "High R (trust model too much)":  "orange",
    "Balanced (optimal)":             "purple",
}

for idx, (name, estimates) in enumerate(results.items()):
    ax = axes[idx]
    ax.plot(ground_truth, color="black", linewidth=2,
            linestyle="--", label="Ground Truth", alpha=0.7)
    ax.plot(estimates, color=colors[name], linewidth=2, label=name)
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="Threshold")
    ax.fill_between(range(N), ground_truth, alpha=0.1, color="black")
    ax.set_ylabel("Occupancy Score")
    ax.set_title(f"{name}\n({configs[name]['desc']})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.2)

axes[2].set_xlabel("Time Step (seconds)")
plt.suptitle("Kalman Filter Q/R Tuning: Effect on Occupancy Estimation",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plot_path = os.path.expanduser("~/Task_6_Sensor_Fusion/fusion_plot.png")
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved → {plot_path}")

# ── Raw sensors plot ──────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 8))
axes2[0].plot(pir_readings, color="orange", alpha=0.8, label="PIR (binary)")
axes2[0].plot(ground_truth, color="black", linestyle="--", linewidth=2, label="Ground Truth")
axes2[0].set_title("PIR Sensor (15% false trigger rate)")
axes2[0].legend(); axes2[0].grid(True, alpha=0.3)

axes2[1].plot(temp_readings, color="green", alpha=0.8, label="Temperature (raw °C drift)")
axes2[1].plot(ground_truth, color="black", linestyle="--", linewidth=2, label="Ground Truth")
axes2[1].set_title("Temperature Sensor (0.5°C Gaussian drift when occupied)")
axes2[1].legend(); axes2[1].grid(True, alpha=0.3)

axes2[2].plot(camera_readings, color="steelblue", alpha=0.8, label="Camera score (0-1)")
axes2[2].plot(ground_truth, color="black", linestyle="--", linewidth=2, label="Ground Truth")
axes2[2].set_title("Camera Motion Score (noisy float 0-1)")
axes2[2].set_xlabel("Time Step (seconds)")
axes2[2].legend(); axes2[2].grid(True, alpha=0.3)

plt.suptitle("Raw Sensor Signals vs Ground Truth", fontsize=13, fontweight="bold")
plt.tight_layout()
raw_plot_path = os.path.expanduser("~/Task_6_Sensor_Fusion/raw_sensors_plot.png")
plt.savefig(raw_plot_path, dpi=150)
print(f"Raw sensors plot saved → {raw_plot_path}")
print("\nDone!")
