


import itertools
import sys
import warnings
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# --------------------------------------------------
# Project paths
# --------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]
src_path = project_root / "src"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(script_dir))

from run_agmm_experiment import experiment

warnings.simplefilter("ignore", category=UserWarning)


def main():
    device = "cpu"
    VERBOSE = False

    print("Running FINAL AGMM convergence experiment on", device)

    # --------------------------------------------------
    # Final thesis settings
    # --------------------------------------------------
    tau_fns = ["sin", "abs"]
    iv_strengths = [0.6]
    estimators = ["AGMM"]
    dgps = ["z_image"]
    num_datas = [2000]
    monte_carlo = 5

    output_dir = project_root / "Analysis" / "tau_comparison_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    performance_records = []
    loss_records = []
    failed_records = []

    settings = list(itertools.product(
        tau_fns, iv_strengths, dgps, num_datas, estimators
    ))

    for tau_fn, iv_strength, dgp, num_data, est in settings:
        print("\n" + "=" * 70)
        print("Settings")
        print("=" * 70)
        print("tau_fn:", tau_fn)
        print("iv_strength:", iv_strength)
        print("dgp:", dgp)
        print("num_data:", num_data)
        print("estimator:", est)

        for run in range(1, monte_carlo + 1):
            print("\n" + "-" * 50)
            print(f"Run {run} of {monte_carlo}")
            print("-" * 50)

            try:
                output = experiment(
                    dgp=dgp,
                    iv_strength=iv_strength,
                    tau_fn=tau_fn,
                    num_data=num_data,
                    est=est,
                    device=device,
                    DEBUG=VERBOSE,
                    return_history=True
                )

                if isinstance(output, tuple) and len(output) == 2:
                    result, loss_history = output
                else:
                    result = output
                    loss_history = None

                print("\nPerformance result:")
                print(result)

                # --------------------------------------------------
                # Save performance result
                # --------------------------------------------------
                if isinstance(result, dict):
                    record = {
                        "tau_fn": tau_fn,
                        "iv_strength": iv_strength,
                        "dgp": dgp,
                        "num_data": num_data,
                        "estimator": est,
                        "run": run,
                        **result,
                    }
                else:
                    result = list(result)

                    record = {
                        "tau_fn": tau_fn,
                        "iv_strength": iv_strength,
                        "dgp": dgp,
                        "num_data": num_data,
                        "estimator": est,
                        "run": run,
                        "R2avg": result[0],
                        "R2fin": result[1],
                        "R2earlystop": result[2],
                        "MSEavg": result[3],
                        "MSEfin": result[4],
                        "MSEearlystop": result[5],
                    }

                performance_records.append(record)

                # --------------------------------------------------
                # Save convergence history
                # --------------------------------------------------
                if loss_history is not None and isinstance(loss_history, dict):
                    if (
                        "epoch" in loss_history
                        and "moment_loss" in loss_history
                        and len(loss_history["epoch"]) > 0
                        and len(loss_history["moment_loss"]) > 0
                    ):
                        loss_df = pd.DataFrame(loss_history)

                        if "epoch" not in loss_df.columns:
                            loss_df["epoch"] = np.arange(1, len(loss_df) + 1)

                        loss_df["tau_fn"] = tau_fn
                        loss_df["iv_strength"] = iv_strength
                        loss_df["dgp"] = dgp
                        loss_df["num_data"] = num_data
                        loss_df["estimator"] = est
                        loss_df["run"] = run

                        # Useful for convergence plots because raw moment values can oscillate around zero
                        if "moment_loss" in loss_df.columns:
                            loss_df["abs_moment_loss"] = loss_df["moment_loss"].abs()

                        loss_records.append(loss_df)

                        print("Saved loss history for this run.")
                        print("Number of epochs recorded:", len(loss_df))

                    else:
                        print("Loss history was returned, but it is empty.")
                else:
                    print("No loss history returned for this run.")

            except Exception as e:
                print("\nERROR in run:")
                print("tau_fn:", tau_fn)
                print("iv_strength:", iv_strength)
                print("dgp:", dgp)
                print("num_data:", num_data)
                print("estimator:", est)
                print("run:", run)
                print("Error message:", str(e))

                failed_records.append({
                    "tau_fn": tau_fn,
                    "iv_strength": iv_strength,
                    "dgp": dgp,
                    "num_data": num_data,
                    "estimator": est,
                    "run": run,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

    # --------------------------------------------------
    # Save raw performance results
    # --------------------------------------------------
    raw_df = pd.DataFrame(performance_records)

    raw_path = output_dir / "tau_comparison_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    print("\nSaved raw performance file:")
    print(raw_path)

    # --------------------------------------------------
    # Save summary performance results
    # --------------------------------------------------
    if not raw_df.empty:
        summary_df = (
            raw_df
            .groupby(["tau_fn", "iv_strength", "dgp", "num_data", "estimator"])
            .agg(
                avg_MSEearlystop=("MSEearlystop", "mean"),
                std_MSEearlystop=("MSEearlystop", "std"),
                avg_MSEfin=("MSEfin", "mean"),
                std_MSEfin=("MSEfin", "std"),
                avg_R2earlystop=("R2earlystop", "mean"),
                std_R2earlystop=("R2earlystop", "std"),
                avg_R2fin=("R2fin", "mean"),
                std_R2fin=("R2fin", "std"),
                MC_successful_runs=("run", "count"),
            )
            .reset_index()
        )

        summary_path = output_dir / "tau_comparison_summary_results.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\nSaved summary performance file:")
        print(summary_path)

        print("\nSummary:")
        print(summary_df)

    # --------------------------------------------------
    # Save failed runs if any
    # --------------------------------------------------
    if failed_records:
        failed_df = pd.DataFrame(failed_records)
        failed_path = output_dir / "failed_runs.csv"
        failed_df.to_csv(failed_path, index=False)

        print("\nSome runs failed. Saved failed-run log:")
        print(failed_path)

    # --------------------------------------------------
    # Save and plot convergence behavior
    # --------------------------------------------------
    if loss_records:
        all_loss_df = pd.concat(loss_records, ignore_index=True)

        loss_raw_path = output_dir / "convergence_loss_raw_results.csv"
        all_loss_df.to_csv(loss_raw_path, index=False)

        print("\nSaved raw convergence file:")
        print(loss_raw_path)

        # Average over Monte Carlo runs
        convergence_summary = (
            all_loss_df
            .groupby(["tau_fn", "iv_strength", "dgp", "num_data", "estimator", "epoch"], as_index=False)
            .agg(
                avg_moment_loss=("moment_loss", "mean"),
                std_moment_loss=("moment_loss", "std"),
                avg_abs_moment_loss=("abs_moment_loss", "mean"),
                std_abs_moment_loss=("abs_moment_loss", "std"),
                MC_runs=("run", "count"),
            )
        )

        convergence_summary_path = output_dir / "convergence_loss_summary_results.csv"
        convergence_summary.to_csv(convergence_summary_path, index=False)

        print("\nSaved convergence summary file:")
        print(convergence_summary_path)

        # --------------------------------------------------
        # Plot 1: separate convergence plot for each tau function
        # --------------------------------------------------
        for tau in tau_fns:
            for iv_strength in iv_strengths:
                sub = convergence_summary[
                    (convergence_summary["tau_fn"] == tau)
                    & (convergence_summary["iv_strength"] == iv_strength)
                ].sort_values("epoch")

                if sub.empty:
                    continue

                plt.figure(figsize=(7, 4.5))
                plt.plot(
                    sub["epoch"],
                    sub["avg_abs_moment_loss"],
                    linewidth=2,
                    label="Average absolute moment loss"
                )

                plt.xlabel("Epoch")
                plt.ylabel("Average absolute moment loss")
                plt.title(f"Convergence behavior: {tau}, IV strength = {iv_strength}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_path = output_dir / f"convergence_behavior_{tau}_iv_{iv_strength}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                print("Saved plot:", plot_path)

        # --------------------------------------------------
        # Plot 2: combined convergence plot for all tau functions
        # --------------------------------------------------
        plt.figure(figsize=(8, 5))

        for tau in tau_fns:
            sub = convergence_summary[
                convergence_summary["tau_fn"] == tau
            ].sort_values("epoch")

            if sub.empty:
                continue

            plt.plot(
                sub["epoch"],
                sub["avg_abs_moment_loss"],
                linewidth=2,
                label=tau
            )

        plt.xlabel("Epoch")
        plt.ylabel("Average absolute moment loss")
        plt.title("AGMM convergence behavior across structural functions")
        plt.legend(title="Tau function")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        combined_plot_path = output_dir / "convergence_behavior_all_tau.png"
        plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved combined convergence plot:", combined_plot_path)

    else:
        print("\nNo convergence loss history was saved.")
        print("Check that agmm_earlystop.py stores self.history during training.")


if __name__ == "__main__":
    main()