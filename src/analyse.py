"""Summarise the control ladder: per-condition accuracy, and the contrast that matters.

The headline number is `connectome - degree_preserving`. If the connectome's
advantage is real topology and not just its degree sequence, that gap is positive
and larger than the spread across seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

LABEL = {
    "connectome": "Connectome (real wiring)",
    "degree_preserving": "Degree-preserving rewire",
    "erdos_renyi": "Erdos-Renyi random",
    "weight_shuffle": "Weights shuffled",
    "no_recurrence": "No recurrence (sanity floor)",
}
ORDER = ["connectome", "degree_preserving", "weight_shuffle", "erdos_renyi", "no_recurrence"]


def load(name: str = "results.json") -> list[dict]:
    return json.loads((RESULTS / name).read_text())


def summarise(runs: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cond in ORDER:
        accs = [r["final_acc"] for r in runs if r["condition"] == cond]
        if not accs:
            continue
        a = np.array(accs)
        out[cond] = {
            "n": len(a),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()),
            "max": float(a.max()),
            "accs": accs,
        }
    return out


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test, returning (t, approximate two-sided p)."""
    from scipy import stats

    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def welch_ci(a: list[float], b: list[float], conf: float = 0.95) -> tuple[float, float, float]:
    """(difference, lo, hi) in percentage points, Welch-corrected.

    The interval is the honest summary: a p-value only says whether zero is
    excluded, while the interval also says how large an effect is still compatible
    with the data. At n=5 these are wide, and saying so is part of the result.
    """
    from scipy import stats

    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff = a.mean() - b.mean()
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    t = stats.t.ppf(1 - (1 - conf) / 2, df)
    return diff * 100, (diff - t * se) * 100, (diff + t * se) * 100


def min_detectable_d(n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest Cohen's d this sample size can find, normal approximation."""
    from scipy import stats

    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(2.0 / n))


def report(name: str = "results.json") -> dict:
    runs = load(name)
    s = summarise(runs)
    print(f"\n{'condition':<28} {'n':>2}  {'mean':>7}  {'std':>6}  {'min':>6}  {'max':>6}")
    print("-" * 66)
    for cond in ORDER:
        if cond not in s:
            continue
        d = s[cond]
        print(
            f"{LABEL[cond]:<28} {d['n']:>2}  {d['mean']:>7.4f}  {d['std']:>6.4f}"
            f"  {d['min']:>6.4f}  {d['max']:>6.4f}"
        )

    # Every null model, tested against the real wiring.
    if "connectome" in s:
        base = s["connectome"]["accs"]
        print(f"\n{'connectome vs ...':<30} {'gap (pp)':>9} {'95% CI':>18} {'p':>8}   verdict")
        print("-" * 84)
        for cond in ORDER:
            if cond == "connectome" or cond not in s:
                continue
            other = s[cond]["accs"]
            gap, lo, hi = welch_ci(base, other)
            _, p = welch(base, other)
            mark = "connectome wins" if p < 0.05 and gap > 0 else (
                "connectome LOSES" if p < 0.05 and gap < 0 else
                ("suggestive" if p < 0.10 else "no detectable difference"))
            print(f"{LABEL[cond]:<30} {gap:>+9.3f} [{lo:>+7.3f},{hi:>+7.3f}] {p:>8.4f}   {mark}")

        n = s["connectome"]["n"]
        print(f"\n  power note: n={n} per condition resolves d >= {min_detectable_d(n):.2f} "
              f"at 80% power. Smaller true effects are not ruled out by a null result.")

    verdict = {}
    if "connectome" in s and "degree_preserving" in s:
        c, d = s["connectome"]["accs"], s["degree_preserving"]["accs"]
        gap = s["connectome"]["mean"] - s["degree_preserving"]["mean"]
        t, p = welch(c, d)
        pooled = np.sqrt((np.var(c, ddof=1) + np.var(d, ddof=1)) / 2)
        print("\n" + "=" * 66)
        print("HEADLINE: connectome vs degree-preserving rewire")
        print("=" * 66)
        print(f"  gap            {gap:+.4f}  ({gap * 100:+.2f} percentage points)")
        print(f"  Welch t        t={t:.3f}   p={p:.4f}")
        print(f"  Cohen's d      {gap / pooled if pooled else float('nan'):.3f}")
        sig = p < 0.05
        print(
            f"\n  VERDICT: {'real wiring WINS' if sig and gap > 0 else ('rewire wins' if sig and gap < 0 else 'NO significant difference')}"
        )
        verdict = {"gap": gap, "t": t, "p": p, "significant": bool(sig)}

    return {"summary": s, "verdict": verdict}


def plot(name: str = "results.json", out: str = "control_ladder.png", task: str = "MNIST") -> None:
    """Two panels: the comparison that matters, and the floor that makes it meaningful."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summarise(load(name))
    net = [c for c in ORDER if c in s and c != "no_recurrence"]
    allc = [c for c in ORDER if c in s]

    def colour(c):
        return "#c2410c" if c == "connectome" else ("#cbd5e1" if c == "no_recurrence" else "#94a3b8")

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    # --- Left: zoomed on the networks, where the claimed effect should live ---
    means = [s[c]["mean"] * 100 for c in net]
    stds = [s[c]["std"] * 100 for c in net]
    ax.bar(range(len(net)), means, yerr=stds, capsize=6,
           color=[colour(c) for c in net], width=0.6, zorder=2)
    for i, c in enumerate(net):
        ax.scatter([i] * len(s[c]["accs"]), [a * 100 for a in s[c]["accs"]],
                   color="#1e293b", zorder=4, s=22, alpha=0.85)
    ax.set_xticks(range(len(net)))
    ax.set_xticklabels([LABEL[c].replace(" (", "\n(") for c in net], fontsize=9)
    ax.set_ylabel(f"{task} test accuracy (%)")
    lo = min(m - sd for m, sd in zip(means, stds)) - 0.15
    hi = max(m + sd for m, sd in zip(means, stds)) + 0.15
    ax.set_ylim(lo, hi)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.02, f"{m:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    if "connectome" in s and "degree_preserving" in s:
        _, p = welch(s["connectome"]["accs"], s["degree_preserving"]["accs"])
        gap = (s["connectome"]["mean"] - s["degree_preserving"]["mean"]) * 100
        if p < 0.05:
            head = "real wiring beats its shuffle" if gap > 0 else "real wiring LOSES to its shuffle"
        elif p < 0.10:
            head = "suggestive, not significant"
        else:
            head = "every network scores the same"
        ax.set_title(
            f"Zoomed: {head}\n"
            f"connectome vs degree-preserving rewire: {gap:+.2f} pp, p = {p:.3f}",
            fontsize=11,
        )

    # --- Right: the floor, proving the network is load-bearing at all ---
    means2 = [s[c]["mean"] * 100 for c in allc]
    ax2.bar(range(len(allc)), means2, color=[colour(c) for c in allc], width=0.65, zorder=2)
    ax2.set_xticks(range(len(allc)))
    ax2.set_xticklabels(["real", "deg", "wts", "ER", "none"][: len(allc)], fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    floor_note = ("the network matters,\nits structure barely does"
                  if "no_recurrence" in s else "all conditions")
    ax2.set_title(f"Full scale: {floor_note}", fontsize=11)
    for i, m in enumerate(means2):
        ax2.text(i, m + 1.5, f"{m:.1f}", ha="center", va="bottom", fontsize=8.5)

    nseeds = ", ".join(sorted({str(s[c]["n"]) for c in allc}))
    fig.suptitle(
        "Does the larval Drosophila connectome beat its own shuffles?   "
        f"frozen recurrent matrix - spectral radius pinned at 0.95 - shared init - {nseeds} seeds",
        fontsize=11.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS / out, dpi=160)
    print(f"wrote {RESULTS / out}")


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    report(name)
    plot(name)
