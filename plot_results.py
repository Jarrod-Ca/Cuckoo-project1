"""plot_results.py - figures for the cuckoo hashing empirical study.

Reads the CSVs the harness wrote to results/ and saves PNGs to results/figures/.
Also prints the summary numbers the report quotes, at full precision.

Run from the repository root:
    python plot_results.py
Needs pandas and matplotlib (both included in Anaconda).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
FIGURES = RESULTS / "figures"

# Theoretical thresholds drawn as reference lines. VERIFY AGAINST A PRIMARY
# SOURCE BEFORE THE REPORT CITES THEM (see ai-use-log.md, Entry 1).
THEORY = {1: 0.5, 4: 0.977}

# One colour per structure, fixed, so a structure is the same colour in every
# figure. Each also has its own marker so colour is never the only cue.
STYLE = {
    "cuckoo_d2b1": dict(color="#2a78d6", marker="o", label="cuckoo d=2, b=1"),
    "cuckoo_d2b4": dict(color="#1baf7a", marker="s", label="cuckoo d=2, b=4"),
    "baseline": dict(color="#eb6834", marker="^", label="std::unordered_map"),
}
STAT_STYLE = {
    "mean": dict(color="#2a78d6", marker="o", label="mean"),
    "p99": dict(color="#eb6834", marker="s", label="p99"),
    "max": dict(color="#1baf7a", marker="^", label="max"),
}
INK = "#2b2b2b"
MUTED = "#8a8a85"
GRID = "#e4e4e0"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
    })


def read_run_info():
    info = {}
    path = RESULTS / "run_info.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    return info


def timer_resolution_ns(lat):
    """Smallest gap between distinct percentile values: the timer's step."""
    values = np.unique(lat[["p50_ns", "p90_ns", "p99_ns", "p999_ns"]].to_numpy().ravel())
    gaps = np.diff(values)
    gaps = gaps[gaps > 1e-6]
    return float(gaps.min()) if gaps.size else float("nan")


# ---------------------------------------------------------------------------
# Figure 1: where insertion first fails, per budget
# ---------------------------------------------------------------------------
def fig_threshold(fails):
    bs = sorted(fails["b"].unique())
    fig, axes = plt.subplots(1, len(bs), figsize=(3.4 * len(bs), 3.2), squeeze=False)
    rng = np.random.default_rng(0)  # jitter only, not data
    for ax, b in zip(axes[0], bs):
        sub = fails[fails["b"] == b]
        mults = sorted(sub["budget_mult"].unique())
        for i, m in enumerate(mults):
            y = sub[sub["budget_mult"] == m]["failure_load"].to_numpy()
            x = i + rng.uniform(-0.12, 0.12, size=y.size)
            ax.scatter(x, y, s=12, color="#2a78d6", alpha=0.55, linewidths=0)
            ax.plot([i - 0.25, i + 0.25], [np.median(y)] * 2, color=INK, linewidth=2)
        budgets = [int(sub[sub["budget_mult"] == m]["budget"].iloc[0]) for m in mults]
        ax.set_xticks(range(len(mults)))
        ax.set_xticklabels([f"{m}x\n({bud})" for m, bud in zip(mults, budgets)])
        ax.set_xlim(-0.6, len(mults) - 0.4)
        if b in THEORY:
            ax.axhline(THEORY[b], color=MUTED, linestyle="--", linewidth=1)
            ax.annotate(f"theory ~{THEORY[b]}", xy=(1.0, THEORY[b]), xycoords=("axes fraction", "data"),
                        xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom", zorder=5,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1),
                        color=MUTED, fontsize=8)
        d = int(sub["d"].iloc[0])
        ax.set_title(f"d={d}, b={b}")
        ax.set_xlabel("eviction budget: multiple of default (evictions)")
    axes[0][0].set_ylabel("load factor at first failed insert")
    seeds = fails["seed"].nunique()
    slots = int(fails["slots"].iloc[0])
    fig.suptitle(f"Insertion failure point: {seeds} seeds per column, {slots:,} slots "
                 "(dots = seeds, bar = median)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_threshold.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: eviction chain length as load rises (largest budget per config,
# so the budget truncates as few walks as possible)
# ---------------------------------------------------------------------------
def fig_chains(chains):
    bs = sorted(chains["b"].unique())
    fig, axes = plt.subplots(1, len(bs), figsize=(3.6 * len(bs), 3.2), squeeze=False)
    for ax, b in zip(axes[0], bs):
        sub = chains[chains["b"] == b]
        top = sub["budget_mult"].max()
        sub = sub[sub["budget_mult"] == top]
        # Median across seeds of each per-bin statistic. Bins near the failure
        # point only contain seeds that got that far.
        agg = sub.groupby("load_bin")[["mean", "p99", "max"]].median()
        for stat, style in STAT_STYLE.items():
            ax.plot(agg.index, agg[stat], color=style["color"], marker=style["marker"],
                    label=style["label"], markevery=max(1, len(agg) // 12))
        budget = int(sub["budget"].iloc[0])
        ax.axhline(budget, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"budget {budget}", xy=(0.0, budget), xycoords=("axes fraction", "data"),
                    xytext=(3, 3), textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
        d = int(sub["d"].iloc[0])
        ax.set_title(f"d={d}, b={b} (budget {top}x)")
        ax.set_xlabel("load factor before the insert")
    axes[0][0].set_ylabel("evictions per insert (symlog)")
    axes[0][-1].legend(loc="center left")
    fig.suptitle("Eviction chain length per 0.01 load bin, median across seeds",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_chain_length.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: lookup latency percentiles against load
# ---------------------------------------------------------------------------
def fig_latency(lat, info, resolution):
    cols = [("p50_ns", "p50"), ("p99_ns", "p99"), ("p999_ns", "p99.9")]
    kinds = [("hit", "successful lookups"), ("miss", "unsuccessful lookups")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
    overhead = float(info.get("timer_overhead_p50_ns", "nan"))
    for r, (kind, kind_label) in enumerate(kinds):
        for c, (col, col_label) in enumerate(cols):
            ax = axes[r][c]
            for name, style in STYLE.items():
                sub = lat[(lat["structure"] == name) & (lat["kind"] == kind)]
                if sub.empty:
                    continue
                g = sub.groupby("load")[col]
                med, lo, hi = g.median(), g.min(), g.max()
                ax.plot(med.index, med.values, color=style["color"], marker=style["marker"],
                        label=style["label"])
                ax.fill_between(med.index, lo.values, hi.values, color=style["color"],
                                alpha=0.15, linewidth=0)
            if not np.isnan(overhead):
                ax.axhline(overhead, color=MUTED, linestyle=":", linewidth=1)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{col_label}, {kind_label}")
            if r == 1:
                ax.set_xlabel("load factor")
            if c == 0:
                ax.set_ylabel("latency (ns)")
    axes[0][0].legend(loc="upper left")
    note = (f"Line = median of trials, band = min to max. Dotted line = timer overhead "
            f"({overhead:.0f} ns), included in every value. Timer resolution on this machine "
            f"~{resolution:.0f} ns, so values are in steps of that size.")
    fig.suptitle("Lookup latency per operation vs load", fontsize=10, color=INK, fontweight="bold")
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=7.5, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "fig3_latency_percentiles.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: probe counts (machine-independent) next to batch throughput
# ---------------------------------------------------------------------------
def fig_probes_vs_time(lat):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6), sharex=True)
    rows = [("mean_probes", "mean slots/nodes examined"), ("batch_mean_ns", "batch mean (ns per lookup)")]
    kinds = [("hit", "successful"), ("miss", "unsuccessful")]
    for r, (col, ylabel) in enumerate(rows):
        for c, (kind, kind_label) in enumerate(kinds):
            ax = axes[r][c]
            for name, style in STYLE.items():
                sub = lat[(lat["structure"] == name) & (lat["kind"] == kind)]
                if sub.empty:
                    continue
                med = sub.groupby("load")[col].median()
                ax.plot(med.index, med.values, color=style["color"], marker=style["marker"],
                        label=style["label"])
            ax.set_ylim(bottom=0)
            ax.set_title(f"{kind_label} lookups")
            if r == 1:
                ax.set_xlabel("load factor")
            if c == 0:
                ax.set_ylabel(ylabel)
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Work per lookup (top) vs time per lookup in a tight loop (bottom)",
                 fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_probes_vs_time.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Numbers for the report
# ---------------------------------------------------------------------------
def print_summary(fails, lat, info, resolution):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 6)
    print("Run:", info.get("timestamp", "?"), "|", info.get("cpu", "?"), "|",
          info.get("compiler", "?"), "|", info.get("build", "?"), "|", info.get("mode", "?"))
    if "QUICK" in info.get("mode", ""):
        print("WARNING: these results come from a --quick smoke test, not a full run.")
    print(f"Detected timer resolution: {resolution:.6f} ns")

    print("\nThreshold experiment: load factor at first failed insert")
    t = fails.groupby(["d", "b", "budget", "budget_mult"])["failure_load"].agg(
        ["count", "mean", "median", "std", "min", "max"])
    print(t)

    print("\nLatency experiment: median over trials (ns), selected loads")
    keep = lat[lat["load"].isin([0.1, 0.45, 0.9, 0.95])]
    l = keep.groupby(["kind", "structure", "load"])[
        ["p50_ns", "p99_ns", "p999_ns", "batch_mean_ns", "mean_probes"]].median()
    print(l)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()
    info = read_run_info()
    fails = pd.read_csv(RESULTS / "threshold_failures.csv")
    chains = pd.read_csv(RESULTS / "threshold_chains.csv")
    lat = pd.read_csv(RESULTS / "latency.csv")
    resolution = timer_resolution_ns(lat)

    fig_threshold(fails)
    fig_chains(chains)
    fig_latency(lat, info, resolution)
    fig_probes_vs_time(lat)
    print_summary(fails, lat, info, resolution)
    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()
