"""Generate the PDF-only figure set for the early-exit paper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports/paper/figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#666666"


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    FIGURES.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_manifest() -> None:
    manifest_path = ROOT / "reports/paper/evidence_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run build_early_exit_paper_tables.py before plotting")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figure_generator"] = str(Path(__file__).relative_to(ROOT))
    audit_path = ROOT / "reports/paper/claim_to_evidence.md"
    supplementary_outputs = {str(path.relative_to(ROOT)) for path in FIGURES.glob("*.pdf")}
    if audit_path.is_file():
        supplementary_outputs.add(str(audit_path.relative_to(ROOT)))
    manifest["outputs"] = sorted(
        set(manifest["outputs"])
        | supplementary_outputs
    )
    manifest["output_sha256"] = {
        path: sha256(ROOT / path) for path in manifest["outputs"]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str, *, dashed: bool = False) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor="white",
        edgecolor=color,
        linewidth=1.35,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#222222")


def arrow(
    ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GRAY, *, dashed: bool = False
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            linestyle="--" if dashed else "-",
            color=color,
        )
    )


def method_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.03, 0.57, 0.11, 0.14, "Input", BLUE)
    box(ax, 0.19, 0.57, 0.15, 0.14, "MobileNetV2\nblocks 0--8", BLUE)
    box(ax, 0.40, 0.57, 0.13, 0.14, "Exit8 head", ORANGE)
    box(ax, 0.59, 0.57, 0.13, 0.14, "$c_8(x)\\geq\\theta$?", ORANGE)
    box(ax, 0.79, 0.76, 0.17, 0.14, "Early prediction", GREEN)
    box(ax, 0.79, 0.38, 0.17, 0.14, "Remaining blocks\n+ final head", BLUE)
    box(ax, 0.43, 0.37, 0.13, 0.12, "Exit16 head\n(training only)", PURPLE, dashed=True)
    arrow(ax, (0.14, 0.64), (0.19, 0.64))
    arrow(ax, (0.34, 0.64), (0.40, 0.64))
    arrow(ax, (0.53, 0.64), (0.59, 0.64))
    arrow(ax, (0.72, 0.67), (0.79, 0.82), GREEN)
    arrow(ax, (0.72, 0.60), (0.79, 0.45), BLUE)
    arrow(ax, (0.50, 0.57), (0.50, 0.49), PURPLE, dashed=True)
    ax.text(0.745, 0.76, "yes", color=GREEN, ha="center")
    ax.text(0.745, 0.50, "no", color=BLUE, ha="center")

    box(ax, 0.03, 0.08, 0.15, 0.12, "40k training", GRAY)
    box(ax, 0.23, 0.08, 0.16, 0.12, "5k model\nselection", GRAY)
    box(ax, 0.44, 0.08, 0.18, 0.12, "5k policy calibration\nor confirmation", ORANGE)
    box(ax, 0.68, 0.08, 0.13, 0.12, "Lock $\\theta$", ORANGE)
    box(ax, 0.86, 0.08, 0.11, 0.12, "One test", GREEN)
    for x1, x2 in ((0.18, 0.23), (0.39, 0.44), (0.62, 0.68), (0.81, 0.86)):
        arrow(ax, (x1, 0.14), (x2, 0.14))
    ax.text(0.03, 0.91, "Dynamic inference", weight="bold", color="#222222")
    ax.text(0.03, 0.28, "Separated selection and evaluation protocol", weight="bold", color="#222222")
    save(fig, "method_overview.pdf")


def accuracy_compute_tradeoff() -> None:
    p1_selection = load("reports/experiments/2026-09-02-early-exit-p1b/locked_selection.json")
    p1_test = load("reports/experiments/2026-09-02-early-exit-p1b/test_results.json")
    boundary = load("reports/diagnostics/2026-09-03-early-exit-p3-boundary-v2/diagnostic.json")
    p4 = load("reports/experiments/2026-09-03-early-exit-p4-cifar100/confirmation.json")
    p4_test = load("reports/experiments/2026-09-03-early-exit-p4-cifar100-test/test_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), sharey=True)
    ax = axes[0]
    calibration = list(p1_selection["locked_policy"]["calibration_metrics"].values())
    x = 100 * mean(row["expected_cost_fraction"] for row in calibration)
    y = -100 * mean(row["accuracy_drop"] for row in calibration)
    ax.plot([100, x], [0, y], color=BLUE, marker="o", linewidth=1.4, label="Development operating points")
    test_agg = p1_test["aggregate"]
    ax.scatter(
        [100 * (1 - test_agg["mac_saving_fraction"]["mean"])],
        [100 * test_agg["locked_policy_gain_vs_final"]["mean"]],
        marker="*",
        s=80,
        color=ORANGE,
        edgecolor="black",
        linewidth=0.4,
        label="Locked test point",
        zorder=4,
    )
    ax.annotate("$\\theta=0.984$", (x, y), xytext=(4, 12), textcoords="offset points")
    ax.set_title("CIFAR-10")
    ax.set_xlabel("Expected Conv/Linear MACs (% of final)")
    ax.set_ylabel("Accuracy difference vs. final head (pp)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    points = []
    strict = boundary["strict_zero_risk_without_15_percent_floor"]
    points.append((strict["threshold"], strict["source"]))
    seen: set[float] = set()
    for item in boundary["exploratory_relaxations"].values():
        budget = item["post_hoc_budget"]["max_worst_class_drop"]
        if budget not in seen:
            points.append((item["threshold"], item["source"]))
            seen.add(budget)
    points.sort(reverse=True)
    xs = [100 * mean(row["expected_cost_fraction"] for row in point[1]["metrics"].values()) for point in points]
    ys = [-100 * mean(row["accuracy_drop"] for row in point[1]["metrics"].values()) for point in points]
    ax.plot(xs, ys, color=BLUE, marker="o", linewidth=1.4, label="P3 post-hoc development")
    p4_metrics = [row["policy_confirmation_metrics"] for row in p4["seed_results"]]
    p4_x = 100 * mean(row["expected_cost_fraction"] for row in p4_metrics)
    p4_y = -100 * mean(row["accuracy_drop"] for row in p4_metrics)
    ax.scatter([p4_x], [p4_y], marker="D", s=35, color=GREEN, label="P4 independent confirmation", zorder=4)
    agg = p4_test["aggregate"]["all_seeds"]
    ax.scatter(
        [100 * (1 - agg["mac_saving_fraction"]["mean"])],
        [100 * agg["locked_policy_gain_vs_final"]["mean"]],
        marker="*",
        s=80,
        color=ORANGE,
        edgecolor="black",
        linewidth=0.4,
        label="Locked test point",
        zorder=4,
    )
    ax.annotate("$\\theta=0.903$", (p4_x, p4_y), xytext=(5, -18), textcoords="offset points")
    ax.set_title("CIFAR-100")
    ax.set_xlabel("Expected Conv/Linear MACs (% of final)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "accuracy_compute_tradeoff.pdf")


def cross_seed_transfer() -> None:
    p1 = load("reports/experiments/2026-09-02-early-exit-p1b/test_results.json")
    p2 = load("reports/experiments/2026-09-03-early-exit-p2a-transfer/transfer_results.json")
    external = load("reports/experiments/2026-09-03-early-exit-p2-cifar10-1-v6/external_results.json")
    labels = ["P1 source\ntest", "P2 unseen\nretraining", "CIFAR-10.1\nall six"]
    early = [
        100 * p1["aggregate"]["early_route_fraction"]["mean"],
        100 * p2["aggregate"]["transfer_early_fraction_mean"],
        100 * external["aggregate"]["all_seeds"]["early_route_fraction"]["mean"],
    ]
    early_sd = [
        100 * p1["aggregate"]["early_route_fraction"]["sample_std"],
        100 * p2["aggregate"]["transfer_early_fraction_sample_std"],
        100 * external["aggregate"]["all_seeds"]["early_route_fraction"]["sample_std"],
    ]
    saving = [
        100 * p1["aggregate"]["mac_saving_fraction"]["mean"],
        100 * p2["aggregate"]["transfer_mac_saving_mean"],
        100 * external["aggregate"]["all_seeds"]["mac_saving_fraction"]["mean"],
    ]
    saving_sd = [
        100 * p1["aggregate"]["mac_saving_fraction"]["sample_std"],
        100 * p2["aggregate"]["transfer_mac_saving_sample_std"],
        100 * external["aggregate"]["all_seeds"]["mac_saving_fraction"]["sample_std"],
    ]
    x = range(3)
    fig, ax = plt.subplots(figsize=(5.3, 2.8))
    ax.errorbar(x, early, yerr=early_sd, marker="o", color=BLUE, capsize=3, label="Early-route fraction")
    ax.errorbar(x, saving, yerr=saving_sd, marker="s", color=ORANGE, capsize=3, label="MAC saving")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Fraction (%)")
    ax.set_ylim(20, 72)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        1.98,
        22.5,
        "No observed worst-class drop\nagainst the corresponding final head",
        ha="right",
        va="bottom",
        color=GRAY,
    )
    fig.tight_layout()
    save(fig, "cross_seed_transfer.pdf")


def risk_budget_boundary() -> None:
    boundary = load("reports/diagnostics/2026-09-03-early-exit-p3-boundary-v2/diagnostic.json")
    p4 = load("reports/experiments/2026-09-03-early-exit-p4-cifar100/confirmation.json")
    strict = boundary["strict_zero_risk_without_15_percent_floor"]
    budgets = [0.0]
    savings = [100 * min(strict["source"]["minimum_mac_saving"], strict["target"]["minimum_mac_saving"])]
    for item in boundary["exploratory_relaxations"].values():
        budget = 100 * item["post_hoc_budget"]["max_worst_class_drop"]
        if budget not in budgets:
            budgets.append(budget)
            savings.append(100 * min(item["source"]["minimum_mac_saving"], item["target"]["minimum_mac_saving"]))
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.plot(budgets, savings, marker="o", color=ORANGE, label="P3 post-hoc boundary")
    p4_metrics = [row["policy_confirmation_metrics"] for row in p4["seed_results"]]
    ax.scatter(
        [4],
        [100 * min(row["cost_saving_fraction"] for row in p4_metrics)],
        marker="D",
        s=42,
        color=GREEN,
        label="P4 independent confirmation",
        zorder=4,
    )
    ax.axhline(15, color=GRAY, linestyle="--", linewidth=1, label="Pre-registered saving floor")
    ax.set_xlabel("Worst-class empirical drop budget (pp)")
    ax.set_ylabel("Minimum MAC saving across seeds (%)")
    ax.set_xticks([0, 2, 4])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "risk_budget_boundary.pdf")


def latency_profile() -> None:
    profile = load("reports/profiles/2026-09-02-early-exit-p1b-rtx4090d/profile.json")
    seeds = [row["seed"] for row in profile["seed_profiles"]]
    reference = [row["summary"]["reference_final_latency_ms"] for row in profile["seed_profiles"]]
    policy = [row["summary"]["expected_policy_latency_ms"] for row in profile["seed_profiles"]]
    fig, ax = plt.subplots(figsize=(4.7, 2.8))
    offsets = {54: -7, 55: 0, 56: 8}
    for seed, left, right in zip(seeds, reference, policy):
        ax.plot([0, 1], [left, right], color="#999999", linewidth=1)
        ax.scatter([0], [left], color=BLUE, s=28)
        ax.scatter([1], [right], color=ORANGE, s=28)
        ax.annotate(
            f"seed {seed}",
            (1, right),
            xytext=(10, offsets[seed]),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
        )
    aggregate = profile["aggregate"]
    ax.plot(
        [0, 1],
        [aggregate["reference_final_latency_ms_mean"], aggregate["expected_policy_latency_ms_mean"]],
        color="black",
        linewidth=2.1,
        marker="D",
        label="Mean",
    )
    ax.set_xlim(-0.25, 1.4)
    ax.set_xticks([0, 1], ["Final only", "Expected policy"])
    ax.set_ylabel("Batch-1 latency (ms)")
    ax.set_ylim(2.9, 4.75)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(0.5, 2.95, "24.91% mean saving", ha="center", color=GREEN, weight="bold")
    fig.tight_layout()
    save(fig, "latency_profile_rtx4090d.pdf")


def main() -> None:
    configure()
    method_overview()
    accuracy_compute_tradeoff()
    cross_seed_transfer()
    risk_budget_boundary()
    latency_profile()
    update_manifest()
    print(f"Wrote PDF figures to {FIGURES}")


if __name__ == "__main__":
    main()
