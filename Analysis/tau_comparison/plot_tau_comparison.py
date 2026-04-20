import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_tau_comparison_results(results_dir: Path) -> pd.DataFrame:
    summary_files = sorted(results_dir.glob("summary_results_*.csv"))
    raw_files = sorted(results_dir.glob("raw_results_*.csv"))

    if summary_files:
        dfs = [pd.read_csv(path) for path in summary_files]
        df = pd.concat(dfs, ignore_index=True)
    elif raw_files:
        dfs = [pd.read_csv(path) for path in raw_files]
        df = pd.concat(dfs, ignore_index=True)
        if "MSE" in df.columns and "avg_MSE" not in df.columns:
            df = df.rename(columns={"MSE": "avg_MSE"})
        df["std_MSE"] = df.get("std_MSE", 0.0)
    else:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    if "avg_MSE" not in df.columns:
        raise ValueError("Loaded results must contain an 'avg_MSE' column")

    if "std_MSE" not in df.columns:
        df["std_MSE"] = 0.0

    return df


def make_tau_comparison_plot(df: pd.DataFrame, output_path: Path) -> None:
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_df = (
        df.groupby("tau_fn")["avg_MSE"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean")
    )

    x = range(len(plot_df))
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    bars = ax.bar(
        x,
        plot_df["mean"],
        yerr=plot_df["std"].values,
        capsize=8,
        color=colors[: len(plot_df)],
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_title("Tau Function Comparison: AGMM Average MSE", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tau Function", fontsize=12)
    ax.set_ylabel("Average MSE", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["tau_fn"])
    ax.set_ylim(0, plot_df["mean"].max() * 1.3)
    ax.grid(axis="y", alpha=0.4)

    for bar, value in zip(bars, plot_df["mean"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + plot_df["std"].max() * 0.05,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to: {output_path}")


def print_summary_table(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("tau_fn")[["avg_MSE", "std_MSE"]]
        .mean()
        .reset_index()
        .sort_values("avg_MSE")
    )

    print("\nTau Comparison Summary Table")
    print(summary.to_string(index=False, float_format="%.8f"))


def main():
    results_dir = Path(__file__).resolve().parent
    df = load_tau_comparison_results(results_dir)

    print_summary_table(df)

    output_plot = results_dir / "tau_comparison_summary.png"
    make_tau_comparison_plot(df, output_plot)


if __name__ == "__main__":
    main()
