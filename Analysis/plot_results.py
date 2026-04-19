import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def load_results(results_dir: Path) -> pd.DataFrame:
    summary_files = sorted(results_dir.glob("summary_results_*.csv"))
    if summary_files:
        dfs = [pd.read_csv(path) for path in summary_files]
        results_df = pd.concat(dfs, ignore_index=True)
    else:
        raw_files = sorted(results_dir.glob("raw_results_*.csv"))
        if not raw_files:
            raise FileNotFoundError(
                f"No summary_results_*.csv or raw_results_*.csv files found in {results_dir}")
        dfs = [pd.read_csv(path) for path in raw_files]
        results_df = pd.concat(dfs, ignore_index=True)
        if "MSE" in results_df.columns and "avg_MSE" not in results_df.columns:
            results_df = results_df.rename(columns={"MSE": "avg_MSE"})
        results_df["std_MSE"] = results_df.get("std_MSE", 0.0)

    if "avg_MSE" not in results_df.columns:
        raise ValueError("Loaded results must contain an 'avg_MSE' or 'MSE' column.")

    if "std_MSE" not in results_df.columns:
        results_df["std_MSE"] = 0.0

    return results_df


def make_plots(results_df: pd.DataFrame, output_path: Path):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    agmm_data = results_df[results_df["estimator"] == "AGMM"].copy()
    if agmm_data.empty:
        raise ValueError("No AGMM rows found in the loaded results.")

    agmm_data["std_MSE"] = agmm_data.get("std_MSE", 0.0)

    pivot_data = agmm_data.pivot_table(
        values="avg_MSE", index="dgp", columns="iv_strength", aggfunc="mean")
    pivot_data.plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("AGMM: MSE by DGP and IV Strength", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Average MSE")
    axes[0].set_xlabel("DGP")
    axes[0].legend(title="IV Strength", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    sample_size_perf = agmm_data.groupby("num_data")["avg_MSE"].agg(["mean", "std"]).reset_index()
    sample_size_perf = sample_size_perf.sort_values("num_data")
    axes[1].errorbar(
        sample_size_perf["num_data"], sample_size_perf["mean"],
        yerr=sample_size_perf["std"], marker="o", linestyle="-", capsize=5,
        color="darkgreen", markersize=8, linewidth=2)
    axes[1].set_title("AGMM: MSE vs Sample Size", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Average MSE")
    axes[1].set_xlabel("Number of Data Points")
    axes[1].grid(True, alpha=0.3)

    tau_perf = agmm_data.groupby("tau_fn")["avg_MSE"].agg(["mean", "std"]).reset_index()
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    axes[2].bar(
        tau_perf["tau_fn"], tau_perf["mean"], yerr=tau_perf["std"],
        capsize=5, color=colors[: len(tau_perf)], alpha=0.75,
        edgecolor="black", linewidth=1.5)
    axes[2].set_title("AGMM: MSE by Tau Function", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Average MSE")
    axes[2].set_xlabel("Tau Function")
    axes[2].grid(True, alpha=0.3, axis="y")

    estimator_comparison = results_df.groupby("estimator")["avg_MSE"].mean().sort_values()
    bar_colors = ["#2ECC71" if x == "AGMM" else "#95A5A6" for x in estimator_comparison.index]
    estimator_comparison.plot(
        kind="barh", ax=axes[3], color=bar_colors, edgecolor="black", linewidth=1.5)
    axes[3].set_title("Overall Performance: AGMM vs Competitors", fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Average MSE")
    axes[3].grid(True, alpha=0.3, axis="x")

    heatmap_data = pivot_data.copy()
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn_r",
        ax=axes[4],
        cbar_kws={"label": "Average MSE"},
        linewidths=0.5)
    axes[4].set_title("AGMM: Performance Heatmap", fontsize=12, fontweight="bold")
    axes[4].set_ylabel("DGP")
    axes[4].set_xlabel("IV Strength")

    axes[5].axis("off")
    summary_stats = agmm_data[
        ["tau_fn", "dgp", "num_data", "iv_strength", "avg_MSE", "std_MSE"]
    ].drop_duplicates()
    summary_stats = summary_stats.sort_values("avg_MSE")
    table_data = [
        [f"{x:.4f}" if isinstance(x, float) else str(x) for x in row]
        for row in summary_stats.values.tolist()
    ]
    table = axes[5].table(
        cellText=table_data,
        colLabels=["Tau Fn", "DGP", "N", "IV Str", "Avg MSE", "Std MSE"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    axes[5].set_title("AGMM: Detailed Results Summary", fontsize=12, fontweight="bold", pad=18)

    plt.suptitle(
        "AGMM Model: Comprehensive Performance Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ AGMM performance visualization saved to '{output_path.name}'")
    plt.show()


def print_summary(agmm_data: pd.DataFrame):
    print("\n" + "=" * 70)
    print("AGMM DETAILED PERFORMANCE ANALYSIS")
    print("=" * 70)
    print(f"\nTotal Configurations Analyzed: {len(agmm_data)}")
    print(f"\nPerformance Summary by Configuration:")
    print(
        agmm_data[
            ["tau_fn", "dgp", "iv_strength", "num_data", "avg_MSE", "std_MSE"]
        ].to_string(index=False))
    best_config = agmm_data.loc[agmm_data["avg_MSE"].idxmin()]
    worst_config = agmm_data.loc[agmm_data["avg_MSE"].idxmax()]
    print(f"\nBest Performance (Lowest MSE):")
    print(
        f"  Configuration: tau_fn={best_config['tau_fn']}, dgp={best_config['dgp']}, "
        f"iv_strength={best_config['iv_strength']}, num_data={best_config['num_data']}")
    print(
        f"  Average MSE: {best_config['avg_MSE']:.6f} "
        f"(±{best_config['std_MSE']:.6f})")
    print(f"\nWorst Performance (Highest MSE):")
    print(
        f"  Configuration: tau_fn={worst_config['tau_fn']}, dgp={worst_config['dgp']}, "
        f"iv_strength={worst_config['iv_strength']}, num_data={worst_config['num_data']}")
    print(
        f"  Average MSE: {worst_config['avg_MSE']:.6f} "
        f"(±{worst_config['std_MSE']:.6f})")
    print("=" * 70)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "results"
    output_path = script_dir / "agmm_performance_analysis.png"

    results_df = load_results(results_dir)
    make_plots(results_df, output_path)

    agmm_data = results_df[results_df["estimator"] == "AGMM"].copy()
    print_summary(agmm_data)


if __name__ == "__main__":
    main()