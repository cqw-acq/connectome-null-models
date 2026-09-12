"""Run the control ladder.

Every condition at a given seed starts from an identical random initialisation of
the trainable parameters; the only thing that differs is the frozen recurrent
matrix. That isolates topology as the variable under test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from connectome import CONDITIONS, build_matrix, load_edges, neuron_roles
from model import ConnectomeReservoir

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_data(name: str = "mnist", batch: int = 256, subset: int | None = None):
    """Flattened image classification. Returns (train_loader, test_loader, n_features)."""
    from torchvision import datasets

    root = str(ROOT / "data" / "torchvision")

    def pack(xs, ys, limit):
        xs = xs.reshape(len(xs), -1).float()
        xs = (xs - xs.mean()) / xs.std()
        ys = torch.as_tensor(ys)
        if limit:
            xs, ys = xs[:limit], ys[:limit]
        return TensorDataset(xs, ys)

    if name == "mnist":
        tr = datasets.MNIST(root, train=True, download=True)
        te = datasets.MNIST(root, train=False, download=True)
        trd, ted = pack(tr.data, tr.targets, subset), pack(te.data, te.targets, None)
    elif name == "cifar10":
        tr = datasets.CIFAR10(root, train=True, download=True)
        te = datasets.CIFAR10(root, train=False, download=True)
        trd = pack(torch.from_numpy(tr.data), tr.targets, subset)
        ted = pack(torch.from_numpy(te.data), te.targets, None)
    else:
        raise ValueError(name)

    n_features = trd.tensors[0].shape[1]
    return (
        DataLoader(trd, batch_size=batch, shuffle=True, drop_last=True),
        DataLoader(ted, batch_size=512),
        n_features,
    )


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, dev: torch.device) -> float:
    model.eval()
    ok = tot = 0
    for x, y in loader:
        pred = model(x.to(dev)).argmax(1).cpu()
        ok += (pred == y).sum().item()
        tot += len(y)
    return ok / tot


def train_one(
    condition: str, seed: int, epochs: int, steps: int, subset: int | None, lr: float,
    dataset: str = "mnist",
):
    dev = device()
    nodes, edges = load_edges("ad")
    inp_idx, out_idx = neuron_roles(nodes)

    t0 = time.time()
    A = build_matrix(condition, nodes, edges, seed=seed)
    build_s = time.time() - t0

    tr, te, n_features = get_data(dataset, subset=subset)

    # Shared random initialisation: identical for every condition at this seed.
    torch.manual_seed(seed)
    model = ConnectomeReservoir(A, inp_idx, out_idx, n_in=n_features, steps=steps).to(dev)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    lossf = nn.CrossEntropyLoss()

    hist = []
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            opt.zero_grad()
            loss = lossf(model(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
        acc = accuracy(model, te, dev)
        hist.append(acc)
        print(f"  [{condition} s{seed}] epoch {ep + 1}/{epochs}  test_acc={acc:.4f}", flush=True)

    return {
        "condition": condition,
        "seed": seed,
        "dataset": dataset,
        "epochs": epochs,
        "steps": steps,
        "subset": subset,
        "final_acc": hist[-1],
        "best_acc": max(hist),
        "history": hist,
        "build_seconds": round(build_s, 1),
        "n_neurons": int(A.shape[0]),
        "n_edges": int((A != 0).sum()),
        "n_input": int(len(inp_idx)),
        "n_output": int(len(out_idx)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    outfile = RESULTS / args.out
    runs = json.loads(outfile.read_text()) if outfile.exists() else []
    done = {(r["condition"], r["seed"]) for r in runs}

    for seed in args.seeds:
        for cond in args.conditions:
            if (cond, seed) in done:
                print(f"skip {cond} seed={seed} (already done)")
                continue
            print(f"=== {cond}  seed={seed}  [{args.dataset}] ===", flush=True)
            t0 = time.time()
            r = train_one(cond, seed, args.epochs, args.steps, args.subset, args.lr, args.dataset)
            r["wall_seconds"] = round(time.time() - t0, 1)
            runs.append(r)
            outfile.write_text(json.dumps(runs, indent=2))
            print(f"  -> final {r['final_acc']:.4f}  ({r['wall_seconds']}s)", flush=True)

    print(f"\nwrote {outfile}")


if __name__ == "__main__":
    main()
