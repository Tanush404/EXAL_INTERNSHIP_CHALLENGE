import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.expanduser("~/Task_5_Federated_Learning/tiny-imagenet-200/train")
NUM_CLIENTS  = 5
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 2
BATCH_SIZE   = 32
ALPHA        = 0.5
DEVICE       = torch.device("cpu")

# ── Transforms ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Load dataset ────────────────────────────────────────────────────────────
print("Loading dataset...")
full_dataset = ImageFolder(DATA_DIR, transform=transform)
num_classes  = len(full_dataset.classes)
print(f"Classes: {num_classes}, Samples: {len(full_dataset)}")

# ── Dirichlet non-IID split ─────────────────────────────────────────────────
def dirichlet_split(dataset, num_clients, alpha):
    labels = np.array([s[1] for s in dataset.samples])
    num_classes = len(dataset.classes)
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet([alpha] * num_clients)
        splits = (proportions * len(class_idx)).astype(int)
        splits[-1] = len(class_idx) - splits[:-1].sum()
        start = 0
        for i, s in enumerate(splits):
            client_indices[i].extend(class_idx[start:start+s].tolist())
            start += s
    return client_indices

np.random.seed(42)
client_indices = dirichlet_split(full_dataset, NUM_CLIENTS, ALPHA)
# Use only 10% of data to speed up training
client_indices = [idx[:len(idx)//10] for idx in client_indices]
for i, idx in enumerate(client_indices):
    print(f"  Client {i}: {len(idx)} samples")

# ── Model ───────────────────────────────────────────────────────────────────
def get_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model.to(DEVICE)

# ── Train & eval ────────────────────────────────────────────────────────────
def train(model, loader, epochs):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += criterion(out, y).item()
            correct  += (out.argmax(1) == y).sum().item()
            total    += y.size(0)
    return loss_sum / len(loader), correct / total

# ── Get model params as numpy ────────────────────────────────────────────────
def get_params(model):
    return [p.detach().numpy().copy() for p in model.parameters()]

def set_params(model, params):
    for p, v in zip(model.parameters(), params):
        p.data = torch.tensor(v)

# ── FedAvg simulation ───────────────────────────────────────────────────────
print("\nStarting Federated Learning simulation...")
global_model  = get_model()
global_params = get_params(global_model)

round_accuracies = []
comm_log = []

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")
    client_updates = []
    client_sizes   = []
    client_accs    = []

    for cid in range(NUM_CLIENTS):
        # Build client data
        subset  = Subset(full_dataset, client_indices[cid])
        n_train = int(0.8 * len(subset))
        train_loader = DataLoader(Subset(subset, range(n_train)), BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(Subset(subset, range(n_train, len(subset))), BATCH_SIZE)

        # Local training
        model = get_model()
        set_params(model, global_params)
        train(model, train_loader, LOCAL_EPOCHS)

        # Comm cost
        comm_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6
        print(f"  Client {cid} comm cost: {comm_mb:.2f} MB")

        # Evaluate
        loss, acc = evaluate(model, val_loader)
        print(f"  Client {cid} val acc: {acc:.4f}")

        client_updates.append(get_params(model))
        client_sizes.append(n_train)
        client_accs.append(acc)

    # FedAvg aggregation
    total = sum(client_sizes)
    new_params = []
    for layer_idx in range(len(global_params)):
        avg = sum(
            client_updates[i][layer_idx] * (client_sizes[i] / total)
            for i in range(NUM_CLIENTS)
        )
        new_params.append(avg)
    global_params = new_params

    global_acc = float(np.mean(client_accs))
    round_accuracies.append(global_acc)
    comm_cost = sum(
        sum(p.size * 4 for p in upd) for upd in client_updates
    ) / 1e6
    comm_log.append({"round": round_num, "global_acc": global_acc, "comm_mb": comm_cost})
    print(f"Round {round_num} global acc: {global_acc:.4f} | comm: {comm_cost:.1f} MB")

# ── Save CSV ────────────────────────────────────────────────────────────────
csv_path = os.path.expanduser("~/Task_5_Federated_Learning/fl_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["round", "global_acc", "comm_mb"])
    w.writeheader()
    w.writerows(comm_log)
print(f"\nCSV saved → {csv_path}")

# ── Convergence plot ────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(range(1, NUM_ROUNDS+1), round_accuracies, marker="o",
         color="steelblue", linewidth=2, label="FL Global Model")
plt.xlabel("Communication Round")
plt.ylabel("Avg Top-1 Accuracy")
plt.title("Federated Learning Convergence (5 Non-IID Camera Clients)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plot_path = os.path.expanduser("~/Task_5_Federated_Learning/fl_convergence.png")
plt.savefig(plot_path, dpi=150)
print(f"Plot saved → {plot_path}")
print("\nDone!")
