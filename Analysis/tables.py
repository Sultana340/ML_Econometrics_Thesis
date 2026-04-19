import pandas as pd
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


def _format_dataframe(df: pd.DataFrame, float_format: str = "{:.4f}") -> pd.DataFrame:
    formatted = df.copy()
    for col in formatted.select_dtypes(include=["float64", "float32"]).columns:
        formatted[col] = formatted[col].map(lambda x: float_format.format(x) if pd.notna(x) else "")
    return formatted


def save_table(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{filename}.csv"
    tex_path = output_dir / f"{filename}.tex"
    df.to_csv(csv_path, index=False)
    try:
        df.to_latex(tex_path, index=False, float_format="%.4f")
    except Exception:
        # If LaTeX export fails, do not interrupt the table generation.
        pass


def make_tables(results_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator_summary = (
        results_df.groupby("estimator")["avg_MSE"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "avg_MSE_mean",
            "std": "avg_MSE_std",
            "min": "avg_MSE_min",
            "max": "avg_MSE_max",
            "count": "n_configs",
        })
    )

    agmm_data = results_df[results_df["estimator"] == "AGMM"].copy()
    if agmm_data.empty:
        raise ValueError("No AGMM rows found in the loaded results.")

    agmm_by_dgp_iv = (
        agmm_data.groupby(["dgp", "iv_strength"])["avg_MSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_MSE_mean", "std": "avg_MSE_std", "count": "n_configs"})
    )

    agmm_by_tau = (
        agmm_data.groupby("tau_fn")["avg_MSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_MSE_mean", "std": "avg_MSE_std", "count": "n_configs"})
    )

    agmm_by_sample_size = (
        agmm_data.groupby("num_data")["avg_MSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_MSE_mean", "std": "avg_MSE_std", "count": "n_configs"})
        .sort_values("num_data")
    )

    top_configs = agmm_data.sort_values("avg_MSE").head(10)
    bottom_configs = agmm_data.sort_values("avg_MSE", ascending=False).head(10)

    save_table(estimator_summary, output_dir, "estimator_performance_summary")
    save_table(agmm_by_dgp_iv, output_dir, "agmm_performance_by_dgp_iv")
    save_table(agmm_by_tau, output_dir, "agmm_performance_by_tau")
    save_table(agmm_by_sample_size, output_dir, "agmm_performance_by_sample_size")
    save_table(top_configs, output_dir, "agmm_top_10_configs")
    save_table(bottom_configs, output_dir, "agmm_bottom_10_configs")
    save_table(agmm_data, output_dir, "agmm_full_results")

    print(f"✓ Saved summary tables to '{output_dir.relative_to(Path.cwd())}'")


def print_tables(results_df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("RESULTS TABLES SUMMARY")
    print("=" * 72)

    estimator_summary = (
        results_df.groupby("estimator")["avg_MSE"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "avg_MSE_mean",
            "std": "avg_MSE_std",
            "min": "avg_MSE_min",
            "max": "avg_MSE_max",
            "count": "n_configs",
        })
    )
    print("\nEstimator summary:")
    print(estimator_summary.to_string(index=False, float_format="%.4f"))

    agmm_data = results_df[results_df["estimator"] == "AGMM"].copy()
    if agmm_data.empty:
        print("\nNo AGMM data available for detailed tables.")
        return

    print("\nAGMM summary by DGP and IV strength:")
    print(
        agmm_data.groupby(["dgp", "iv_strength"])["avg_MSE"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "avg_MSE_mean", "std": "avg_MSE_std", "count": "n_configs"})
        .reset_index()
        .to_string(index=False, float_format="%.4f")
    )

    print("\nAGMM summary by tau function:")
    print(
        agmm_data.groupby("tau_fn")["avg_MSE"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "avg_MSE_mean", "std": "avg_MSE_std", "count": "n_configs"})
        .reset_index()
        .to_string(index=False, float_format="%.4f")
    )

    print("\nAGMM summary by sample size:")
    print(
        agmm_data.groupby("num_data")["avg_MSE"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "avg_MSE_mean", "std": "avg_MSE_std", "count": "n_configs"})
        .reset_index()
        .sort_values("num_data")
        .to_string(index=False, float_format="%.4f")
    )

    best_config = agmm_data.loc[agmm_data["avg_MSE"].idxmin()]
    worst_config = agmm_data.loc[agmm_data["avg_MSE"].idxmax()]

    print("\nBest AGMM configuration:")
    print(best_config.to_frame().T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nWorst AGMM configuration:")
    print(worst_config.to_frame().T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir.parent / "results"
    output_dir = script_dir / "tables_output"

    results_df = load_results(results_dir)
    make_tables(results_df, output_dir)
    print_tables(results_df)


if __name__ == "__main__":
    main()
