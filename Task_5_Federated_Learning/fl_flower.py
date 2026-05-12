import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from flwr.common import NDArrays, Scalar

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.expanduser("~/Task_5_Federated_Learning/tiny-imagenet-200/train")
NUM_CLIENTS  = 5
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 2
BATCH_SIZE   = 32
ALPHA        = 0.5
DEVICE       = torch.device("cpu")

# ── Transforms ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Dataset ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
full_dataset = ImageFolder(DATA_DIR, transform=transform)
num_classes  = len(full_dataset.classes)
print(f"Classes: {num_classes}, Samples: {len(full_dataset)}")

# ── Dirichlet non-IID split ──────────────────────────────────────────────────
def dirichlet_split(dataset, num_clients, alpha):
    labels = np.array([s[1] for s in dataset.samples])
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
# Use 10% subset for speed
client_indices = [idx[:len(idx)//10] for idx in client_indices]
for i, idx in enumerate(client_indices):
    print(f"  Client {i}: {len(idx)} samples")

# ── Model ────────────────────────────────────────────────────────────────────
def get_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model.to(DEVICE)

def get_params(model):
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]

def set_params(model, params):
    for p, v in zip(model.parameters(), params):
        p.data = torch.tensor(v).to(DEVICE)

# ── Train & eval ─────────────────────────────────────────────────────────────
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

# ── Flower Client ─────────────────────────────────────────────────────────────
class CameraClient(fl.client.NumPyClient):
    """
    Simulates a smart home camera node.
    Each client holds a private non-IID shard of Tiny ImageNet.
    No raw data leaves the device — only model weights are communicated.
    """
    def __init__(self, cid: int, indices: List[int]):
        self.cid   = cid
        subset     = Subset(full_dataset, indices)
        n_train    = int(0.8 * len(subset))
        self.train_loader = DataLoader(
            Subset(subset, range(n_train)), BATCH_SIZE, shuffle=True)
        self.val_loader   = DataLoader(
            Subset(subset, range(n_train, len(subset))), BATCH_SIZE)
        self.model = get_model()

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_params(self.model)

    def fit(self, parameters: NDArrays,
            config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        set_params(self.model, parameters)
        train(self.model, self.train_loader, LOCAL_EPOCHS)
        comm_mb = sum(p.numel() * 4
                      for p in self.model.parameters()) / 1e6
        print(f"    Client {self.cid} trained | comm: {comm_mb:.2f} MB")
        return get_params(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: NDArrays,
                 config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        set_params(self.model, parameters)
        loss, acc = evaluate(self.model, self.val_loader)
        print(f"    Client {self.cid} val acc: {acc:.4f}")
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(acc)}


# ── Flower Server strategy ────────────────────────────────────────────────────
def weighted_average(metrics):
    """Aggregate accuracy from all clients using weighted average."""
    total   = sum(n for n, _ in metrics)
    acc_agg = sum(n * m["accuracy"] for n, m in metrics) / total
    return {"accuracy": acc_agg}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,           # use all clients every round
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# ── Client factory ────────────────────────────────────────────────────────────
def client_fn(cid: str) -> fl.client.Client:
    return CameraClient(int(cid), client_indices[int(cid)]).to_client()

# ── Run simulation ────────────────────────────────────────────────────────────
print("\nStarting Flower FL simulation...")
history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

print("\n--- Round-by-round accuracy ---")
if history.metrics_distributed:
    for round_num, metrics in history.metrics_distributed.get("accuracy", []):
        print(f"  Round {round_num}: {metrics:.4f}")

print("\nFlower simulation complete!")
print("(Use fl_train.py results for final CSV/plots — this script validates Flower API usage)")
