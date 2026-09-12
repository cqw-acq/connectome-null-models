"""Load the Drosophila larva connectome and build the null models it gets compared against.

Data: Winding et al. 2023, "The connectome of an insect brain" (Science), via
netzschleuder (`fly_larva`). 2,956 neurons, 116,922 typed edges.

We keep only the axon->dendrite (`ad`) edges, 63,545 of them, because that is the
canonical feedforward synapse direction and it reproduces the "3,000 neurons /
65,000 weights" figure reported by the Biological Processing Units paper
(arXiv:2507.10951), whose claim this experiment is built to test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

# Sensory neurons are the injection site; descending neurons carry the brain's
# output to the nerve cord and are the readout site.
INPUT_TYPES = ("sensory",)
OUTPUT_TYPES = ("DN-VNC", "DN-SEZ")


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name)
    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def load_edges(etype: str | None = "ad") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nodes, edges). `etype=None` keeps all four synapse types."""
    nodes, edges = _read("nodes.csv"), _read("edges.csv")
    if etype is not None:
        edges = edges[edges.etype == etype].reset_index(drop=True)
    return nodes, edges


def neuron_roles(nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the input (sensory) and output (descending) populations."""
    inp = np.flatnonzero(nodes.cell_type.isin(INPUT_TYPES).values)
    out = np.flatnonzero(nodes.cell_type.isin(OUTPUT_TYPES).values)
    return inp, out


def to_matrix(n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Dense [target, source] matrix, so that `h_next = A @ h`."""
    A = np.zeros((n, n), dtype=np.float32)
    np.add.at(A, (dst, src), w.astype(np.float32))
    return A


# ---------------------------------------------------------------------------
# Null models. Each one destroys a different amount of structure, so that a
# performance gap can be attributed to the thing that was actually removed.
# ---------------------------------------------------------------------------


def degree_preserving_swap(
    src: np.ndarray, dst: np.ndarray, rng: np.random.Generator, n_swaps_per_edge: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Directed double-edge swap: (a->b, c->d) becomes (a->d, c->b).

    Every neuron keeps its exact in-degree and out-degree; only *who wires to whom*
    is randomised. This is the control that matters — it separates "this specific
    wiring pattern is special" from "having this many connections is what helps".
    """
    src, dst = src.copy(), dst.copy()
    m = len(src)
    present = set(zip(src.tolist(), dst.tolist()))
    target_swaps = n_swaps_per_edge * m
    done = 0
    attempts = 0
    max_attempts = target_swaps * 10

    while done < target_swaps and attempts < max_attempts:
        batch = min(20000, target_swaps - done)
        i = rng.integers(0, m, size=batch)
        j = rng.integers(0, m, size=batch)
        attempts += batch
        for x, y in zip(i, j):
            if x == y:
                continue
            a, b, c, d = src[x], dst[x], src[y], dst[y]
            if a == d or c == b:          # would create a self-loop
                continue
            if (a, d) in present or (c, b) in present:   # would duplicate an edge
                continue
            present.discard((a, b))
            present.discard((c, d))
            present.add((a, d))
            present.add((c, b))
            dst[x], dst[y] = d, b
            done += 1
    return src, dst


def erdos_renyi(n: int, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly random directed graph with the same node and edge count."""
    seen: set[tuple[int, int]] = set()
    while len(seen) < m:
        need = m - len(seen)
        a = rng.integers(0, n, size=need * 2)
        b = rng.integers(0, n, size=need * 2)
        for x, y in zip(a.tolist(), b.tolist()):
            if x != y:
                seen.add((x, y))
            if len(seen) == m:
                break
    src, dst = map(np.array, zip(*seen))
    return src, dst


def dale_signs(n: int, rng: np.random.Generator, frac_inhibitory: float = 0.3) -> np.ndarray:
    """One fixed sign per *presynaptic* neuron (Dale's law).

    The larval dataset carries no neurotransmitter labels, so signs are assigned
    at random — but the SAME sign vector is reused across every condition, so it
    can never be the thing that separates them.
    """
    s = np.ones(n, dtype=np.float32)
    s[rng.random(n) < frac_inhibitory] = -1.0
    return s


def spectral_normalise(A: np.ndarray, radius: float = 0.95) -> np.ndarray:
    """Rescale so the largest eigenvalue magnitude equals `radius`.

    This is the control Dhiman (arXiv:2604.04033) named as missing. A reservoir's
    behaviour is dominated by its spectral radius; without pinning it, a
    connectome-vs-shuffle comparison silently compares two different dynamical
    regimes instead of two different topologies.
    """
    eig = np.linalg.eigvals(A.astype(np.float64))
    rho = float(np.abs(eig).max())
    if rho < 1e-12:
        return A
    return (A * (radius / rho)).astype(np.float32)


CONDITIONS = ("connectome", "degree_preserving", "erdos_renyi", "weight_shuffle", "no_recurrence")


def build_matrix(
    condition: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seed: int,
    radius: float = 0.95,
    signed: bool = True,
) -> np.ndarray:
    """Build the frozen recurrent matrix for one rung of the control ladder."""
    rng = np.random.default_rng(seed)
    n = len(nodes)
    src = edges.source.values.astype(np.int64)
    dst = edges.target.values.astype(np.int64)
    w = edges["count"].values.astype(np.float32)

    if condition == "no_recurrence":
        # Sanity floor: no recurrent network at all. If this scores as well as the
        # connectome, the task is being solved by the trainable projections alone
        # and the whole topology comparison is measuring nothing.
        return np.zeros((n, n), dtype=np.float32)
    if condition == "connectome":
        pass
    elif condition == "degree_preserving":
        src, dst = degree_preserving_swap(src, dst, rng)
    elif condition == "erdos_renyi":
        src, dst = erdos_renyi(n, len(src), rng)
        w = rng.permutation(w)
    elif condition == "weight_shuffle":
        w = rng.permutation(w)          # topology intact, synapse strengths scrambled
    else:
        raise ValueError(f"unknown condition: {condition}")

    A = to_matrix(n, src, dst, w)
    if signed:
        # Sign depends on the presynaptic neuron -> multiply columns.
        A = A * dale_signs(n, np.random.default_rng(1234), 0.3)[None, :]
    return spectral_normalise(A, radius)
