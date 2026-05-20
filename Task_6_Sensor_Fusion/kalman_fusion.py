import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ── Kalman Filter from scratch ──────────────────────────────────────────────
class KalmanFilter:
    """
    1D Kalman Filter for occupancy probability estimation.
    State: [occupancy_probability]
    Fuses three sensors: PIR, temperature delta, camera confidence
    """
    def __init__(self):
        # State estimate
        self.x = np.array([[0.0]])      # initial occupancy = 0 (empty room)

        # State covariance — how uncertain we are about the state
        self.P = np.array([[1.0]])

        # State transition matrix — occupancy doesn't change on its own
        self.F = np.array([[1.0]])

        # Process noise — how much the true state can change between steps
        self.Q = np.array([[0.01]])

        # Observation matrix — each sensor directly observes occupancy
        self.H = np.array([[1.0]])

        # Measurement noise covariance per sensor
        self.R_pir    = np.array([[0.3]])   # PIR: noisy (pets trigger it)
        self.R_temp   = np.array([[0.5]])   # Temp: slow to respond
        self.R_camera = np.array([[0.1]])   # Camera: most reliable

    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0]

    def update(self, z, R):
        """Update state with a new measurement z and noise R."""
        # Innovation (difference between measurement and prediction)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain — how much to trust the measurement vs prediction
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        # Clip to valid probability range
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)

        return self.x[0, 0]


# ── Simulate sensor data ────────────────────────────────────────────────────
np.random.seed(42)
N = 100   # 100 time steps

# Ground truth occupancy (1 = occupied, 0 = empty)
ground_truth = np.zeros(N)
ground_truth[20:60] = 1.0   # person in room from t=20 to t=60
ground_truth[75:90] = 1.0   # person returns t=75 to t=90

# Simulate noisy sensor readings
def simulate_pir(gt):
    """PIR: binary motion sensor. High false positive rate (pets)."""
    readings = []
    for g in gt:
        if g == 1.0:
            # True positive rate: 85%
            readings.append(np.random.choice([1.0, 0.0], p=[0.85, 0.15]))
        else:
            # False positive rate: 20% (pets, drafts)
            readings.append(np.random.choice([0.0, 1.0], p=[0.80, 0.20]))
    return np.array(readings)

def simulate_temp(gt):
    """Temperature delta sensor. Slow response, gradual change."""
    readings = []
    for i, g in enumerate(gt):
        if g == 1.0:
            # Body heat raises temp gradually
            val = 0.7 + np.random.normal(0, 0.15)
        else:
            val = 0.1 + np.random.normal(0, 0.1)
        readings.append(np.clip(val, 0, 1))
    return np.array(readings)

def simulate_camera(gt):
    """Camera confidence score from MobileNetV2 (your Week 1-3 model!)."""
    readings = []
    for g in gt:
        if g == 1.0:
            # High confidence when person present
            val = 0.85 + np.random.normal(0, 0.1)
        else:
            # Low confidence when empty (occasional false detection)
            val = 0.05 + np.random.normal(0, 0.05)
        readings.append(np.clip(val, 0, 1))
    return np.array(readings)

pir_readings    = simulate_pir(ground_truth)
temp_readings   = simulate_temp(ground_truth)
camera_readings = simulate_camera(ground_truth)

# ── Run Kalman Filter ───────────────────────────────────────────────────────
kf = KalmanFilter()
fused_estimates = []
raw_pir_only    = []

for t in range(N):
    # Predict
    kf.predict()

    # Update with each sensor
    kf.update(np.array([[pir_readings[t]]]),    kf.R_pir)
    kf.update(np.array([[temp_readings[t]]]),   kf.R_temp)
    kf.update(np.array([[camera_readings[t]]]), kf.R_camera)

    fused_estimates.append(kf.x[0, 0])
    raw_pir_only.append(pir_readings[t])

fused_estimates = np.array(fused_estimates)

# ── Metrics ─────────────────────────────────────────────────────────────────
threshold = 0.5

# Fused system
fused_pred = (fused_estimates > threshold).astype(int)
fused_tp = np.sum((fused_pred == 1) & (ground_truth == 1))
fused_fp = np.sum((fused_pred == 1) & (ground_truth == 0))
fused_tn = np.sum((fused_pred == 0) & (ground_truth == 0))
fused_fn = np.sum((fused_pred == 0) & (ground_truth == 1))
fused_acc     = (fused_tp + fused_tn) / N
fused_far     = fused_fp / (fused_fp + fused_tn) if (fused_fp + fused_tn) > 0 else 0
fused_frr     = fused_fn / (fused_fn + fused_tp) if (fused_fn + fused_tp) > 0 else 0

# PIR only
pir_pred = (np.array(raw_pir_only) > threshold).astype(int)
pir_tp = np.sum((pir_pred == 1) & (ground_truth == 1))
pir_fp = np.sum((pir_pred == 1) & (ground_truth == 0))
pir_tn = np.sum((pir_pred == 0) & (ground_truth == 0))
pir_fn = np.sum((pir_pred == 0) & (ground_truth == 1))
pir_acc = (pir_tp + pir_tn) / N
pir_far = pir_fp / (pir_fp + pir_tn) if (pir_fp + pir_tn) > 0 else 0
pir_frr = pir_fn / (pir_fn + pir_tp) if (pir_fn + pir_tp) > 0 else 0

print("=" * 50)
print("SENSOR FUSION RESULTS")
print("=" * 50)
print(f"\nPIR only:")
print(f"  Accuracy: {pir_acc:.3f}")
print(f"  FAR (False Alarm Rate): {pir_far:.3f}")
print(f"  FRR (False Rejection Rate): {pir_frr:.3f}")
print(f"\nFused (PIR + Temp + Camera):")
print(f"  Accuracy: {fused_acc:.3f}")
print(f"  FAR (False Alarm Rate): {fused_far:.3f}")
print(f"  FRR (False Rejection Rate): {fused_frr:.3f}")

# ── Save CSV ─────────────────────────────────────────────────────────────────
csv_path = os.path.expanduser("~/Task_6_Sensor_Fusion/fusion_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "ground_truth", "pir", "temp", "camera", "fused_estimate"])
    for t in range(N):
        w.writerow([t, ground_truth[t], pir_readings[t],
                    temp_readings[t], camera_readings[t], fused_estimates[t]])
print(f"\nCSV saved → {csv_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Top: raw sensor readings
axes[0].plot(pir_readings,    label="PIR",         alpha=0.7, color="orange")
axes[0].plot(temp_readings,   label="Temperature", alpha=0.7, color="green")
axes[0].plot(camera_readings, label="Camera",      alpha=0.7, color="steelblue")
axes[0].plot(ground_truth,    label="Ground Truth",
             color="black", linewidth=2, linestyle="--")
axes[0].set_ylabel("Sensor Reading")
axes[0].set_title("Raw Sensor Readings vs Ground Truth")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Middle: Kalman fused estimate
axes[1].plot(fused_estimates, label="Kalman Fused",
             color="purple", linewidth=2)
axes[1].plot(ground_truth, label="Ground Truth",
             color="black", linewidth=2, linestyle="--")
axes[1].axhline(0.5, color="red", linestyle=":", label="Decision threshold")
axes[1].set_ylabel("Occupancy Probability")
axes[1].set_title("Kalman Filter Fused Estimate")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Bottom: comparison
axes[2].plot(fused_pred,   label=f"Fused (acc={fused_acc:.2f})",
             color="purple", linewidth=2)
axes[2].plot(pir_pred,     label=f"PIR only (acc={pir_acc:.2f})",
             color="orange", alpha=0.7)
axes[2].plot(ground_truth, label="Ground Truth",
             color="black", linewidth=2, linestyle="--")
axes[2].set_ylabel("Occupancy Decision")
axes[2].set_xlabel("Time Step")
axes[2].set_title("PIR Only vs Fused Decision")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.expanduser("~/Task_6_Sensor_Fusion/fusion_plot.png")
plt.savefig(plot_path, dpi=150)
print(f"Plot saved → {plot_path}")
print("\nDone!")
