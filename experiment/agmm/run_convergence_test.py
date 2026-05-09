


import itertools
import os
import sys
import warnings
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

    print("Running Thesis AGMM experiment on", device)

    tau_fns = ["linear", "sin", "abs"]
    iv_strengths = [0.6]
    estimators = ["AGMM"]
    dgps = ["z_image"]
    num_datas = [2000]

    monte_carlo = 5

    output_dir = project_root / "Analysis" / "tau_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    performance_records = []
    loss_records = []

    settings = list(itertools.product(
        tau_fns, iv_strengths, dgps, num_datas, estimators
    ))

    for tau_fn, iv_strength, dgp, num_data, est in settings:
        print("\n------ Settings ------")
        print("tau_fn:", tau_fn)
        print("iv_strength:", iv_strength)
        print("dgp:", dgp)
        print("num_data:", num_data)
        print("estimator:", est)

        for run in range(1, monte_carlo + 1):
            print("\nRun", run)

            # --------------------------------------------------
            # experiment() should return:
            # result, loss_history
            # when return_history=True
            # --------------------------------------------------
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

            # Keep safe if experiment returns only result
            if isinstance(output, tuple) and len(output) == 2:
                result, loss_history = output
            else:
                result = output
                loss_history = None

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
            # Save convergence losses if available
            # --------------------------------------------------
            if loss_history is not None:
                loss_df = pd.DataFrame(loss_history)

                # If epoch column is missing, create it from row index
                if "epoch" not in loss_df.columns:
                    loss_df["epoch"] = np.arange(1, len(loss_df) + 1)

                loss_df["tau_fn"] = tau_fn
                loss_df["iv_strength"] = iv_strength
                loss_df["dgp"] = dgp
                loss_df["num_data"] = num_data
                loss_df["estimator"] = est
                loss_df["run"] = run

                loss_records.append(loss_df)

    # --------------------------------------------------
    # Save raw performance results
    # --------------------------------------------------
    raw_df = pd.DataFrame(performance_records)

    raw_path = output_dir / "tau_comparison_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    # --------------------------------------------------
    # Save summary performance results
    # --------------------------------------------------
    summary_df = (
        raw_df
        .groupby(["tau_fn", "iv_strength", "dgp", "num_data", "estimator"])
        .agg(
            avg_MSEearlystop=("MSEearlystop", "mean"),
            std_MSEearlystop=("MSEearlystop", "std"),
            avg_R2fin=("R2fin", "mean"),
            std_R2fin=("R2fin", "std"),
            MC_successful_runs=("run", "count"),
        )
        .reset_index()
    )

    summary_path = output_dir / "tau_comparison_summary_results.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved performance files:")
    print(raw_path)
    print(summary_path)

    # --------------------------------------------------
    # Save and plot convergence behavior
    # --------------------------------------------------
    if loss_records:
        all_loss_df = pd.concat(loss_records, ignore_index=True)

        loss_path = output_dir / "convergence_loss_raw_results.csv"
        all_loss_df.to_csv(loss_path, index=False)

        print("\nSaved convergence loss file:")
        print(loss_path)

        # Check which loss columns are available
        available_loss_cols = []

        if "moment_loss" in all_loss_df.columns:
            available_loss_cols.append("moment_loss")

        if "kernel_loss" in all_loss_df.columns:
            available_loss_cols.append("kernel_loss")

        if not available_loss_cols:
            print("\nLoss history exists, but no moment_loss or kernel_loss column was found.")
            print("Available columns are:")
            print(list(all_loss_df.columns))
            return

        # Average over Monte Carlo runs
        agg_dict = {
            f"avg_{col}": (col, "mean")
            for col in available_loss_cols
        }

        convergence_df = (
            all_loss_df
            .groupby(["tau_fn", "iv_strength", "epoch"], as_index=False)
            .agg(**agg_dict)
        )

        convergence_path = output_dir / "convergence_loss_summary_results.csv"
        convergence_df.to_csv(convergence_path, index=False)

        print("\nSaved convergence summary file:")
        print(convergence_path)

        # Plot separately for each tau function and IV strength
        for tau in tau_fns:
            for iv_strength in iv_strengths:
                sub = convergence_df[
                    (convergence_df["tau_fn"] == tau)
                    & (convergence_df["iv_strength"] == iv_strength)
                ].sort_values("epoch")

                if sub.empty:
                    continue

                plt.figure(figsize=(7, 4.5))

                if "avg_moment_loss" in sub.columns:
                    plt.plot(
                        sub["epoch"],
                        sub["avg_moment_loss"],
                        linewidth=2,
                        label="Moment loss"
                    )

                if "avg_kernel_loss" in sub.columns:
                    plt.plot(
                        sub["epoch"],
                        sub["avg_kernel_loss"],
                        linewidth=2,
                        label="Kernel loss"
                    )

                plt.xlabel("Epoch")
                plt.ylabel("Average loss")
                plt.title(f"Convergence behavior: {tau}, IV strength = {iv_strength}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_path = output_dir / f"convergence_behavior_{tau}_iv_{iv_strength}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.show()

                print("Saved plot:", plot_path)

    else:
        print("\nNo convergence loss history was returned by experiment().")
        print("Make sure run_agmm_experiment.py has return_history=True support.")
        print("Also make sure train_agmm or AGMM stores epoch-level moment_loss.")


if __name__ == "__main__":
    main()