import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_path = os.path.join(project_root, "src")

sys.path.insert(0, src_path)
sys.path.insert(0, script_dir)

from run_agmm_experiment import experiment

warnings.simplefilter("ignore", category=UserWarning)


def main():
    device = "cpu"
    print("Running CentroidMMDGMM robustness check on", device)

    # --------------------------------------------------
    # Final thesis robustness setting
    # --------------------------------------------------
    VERBOSE = False

    tau_fn_list = ["abs"]
    iv_strength_list = [0.6]
    dgps = ["z_image"]
    num_data_list = [2000]

    # Keep g_features fixed for CentroidMMDGMM
    g_features = 10

    # Robustness dimension for CentroidMMDGMM
    n_centers_list = [5, 10, 25]

    # Robustness checks can use 3 MC runs to save CPU time
    monte_carlo = 3

    estimator = "CentroidMMDGMM"

    settings = list(itertools.product(
        tau_fn_list,
        iv_strength_list,
        dgps,
        num_data_list,
        n_centers_list
    ))

    all_raw_rows = []
    all_summary_rows = []

    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    raw_path = os.path.join(
        results_dir,
        "robustness_centroid_ncenters_raw_abs_iv06.csv"
    )

    summary_path = os.path.join(
        results_dir,
        "robustness_centroid_ncenters_summary_abs_iv06.csv"
    )

    # --------------------------------------------------
    # Run experiments
    # --------------------------------------------------
    for tau_fn, iv_strength, dgp, num_data, n_centers in settings:
        print("\n" + "=" * 60)
        print("Settings")
        print("=" * 60)
        print("tau_fn:", tau_fn)
        print("iv_strength:", iv_strength)
        print("dgp:", dgp)
        print("num_data:", num_data)
        print("g_features:", g_features)
        print("n_centers:", n_centers)
        print("estimator:", estimator)

        mse_earlystop_results = []
        mse_fin_results = []
        r2_fin_results = []

        for run in range(monte_carlo):
            print("\nRun", run + 1, "of", monte_carlo)

            start_time = time.time()

            try:
                result = experiment(
                    dgp,
                    iv_strength,
                    tau_fn,
                    num_data,
                    "CentroidMMDGMM",
                    device=device,
                    DEBUG=VERBOSE,
                    g_features=10,
                    n_centers=n_centers
                )

                runtime_seconds = time.time() - start_time

                r2_avg = float(result[0])
                r2_fin = float(result[1])
                r2_earlystop = float(result[2])
                mse_avg = float(result[3])
                mse_fin = float(result[4])
                mse_earlystop = float(result[5])

                mse_earlystop_results.append(mse_earlystop)
                mse_fin_results.append(mse_fin)
                r2_fin_results.append(r2_fin)

                all_raw_rows.append({
                    "tau_fn": tau_fn,
                    "iv_strength": iv_strength,
                    "dgp": dgp,
                    "num_data": num_data,
                    "g_features": g_features,
                    "n_centers": n_centers,
                    "estimator": estimator,
                    "run": run + 1,
                    "R2_avg": r2_avg,
                    "R2_fin": r2_fin,
                    "R2_earlystop": r2_earlystop,
                    "MSE_avg": mse_avg,
                    "MSE_fin": mse_fin,
                    "MSE_earlystop": mse_earlystop,
                    "runtime_seconds": runtime_seconds,
                    "error": ""
                })

                pd.DataFrame(all_raw_rows).to_csv(raw_path, index=False)
                print("Saved partial raw results.")
                print("Runtime seconds:", round(runtime_seconds, 2))

            except Exception as e:
                runtime_seconds = time.time() - start_time

                print("\nERROR in setting:")
                print("tau_fn:", tau_fn)
                print("iv_strength:", iv_strength)
                print("dgp:", dgp)
                print("num_data:", num_data)
                print("g_features:", g_features)
                print("n_centers:", n_centers)
                print("estimator:", estimator)
                print("run:", run + 1)
                print("Error message:", e)

                all_raw_rows.append({
                    "tau_fn": tau_fn,
                    "iv_strength": iv_strength,
                    "dgp": dgp,
                    "num_data": num_data,
                    "g_features": g_features,
                    "n_centers": n_centers,
                    "estimator": estimator,
                    "run": run + 1,
                    "R2_avg": np.nan,
                    "R2_fin": np.nan,
                    "R2_earlystop": np.nan,
                    "MSE_avg": np.nan,
                    "MSE_fin": np.nan,
                    "MSE_earlystop": np.nan,
                    "runtime_seconds": runtime_seconds,
                    "error": str(e)
                })

                pd.DataFrame(all_raw_rows).to_csv(raw_path, index=False)
                continue

        # --------------------------------------------------
        # Summary for this n_centers setting
        # --------------------------------------------------
        successful_runs = len(mse_earlystop_results)

        if successful_runs > 0:
            avg_mse_earlystop = np.mean(mse_earlystop_results)
            std_mse_earlystop = np.std(mse_earlystop_results, ddof=1) if successful_runs > 1 else 0.0

            avg_mse_fin = np.mean(mse_fin_results)
            std_mse_fin = np.std(mse_fin_results, ddof=1) if successful_runs > 1 else 0.0

            avg_r2_fin = np.mean(r2_fin_results)
            std_r2_fin = np.std(r2_fin_results, ddof=1) if successful_runs > 1 else 0.0
        else:
            avg_mse_earlystop = np.nan
            std_mse_earlystop = np.nan
            avg_mse_fin = np.nan
            std_mse_fin = np.nan
            avg_r2_fin = np.nan
            std_r2_fin = np.nan

        print("\n---------- Summary ----------")
        print("Successful runs:", successful_runs)
        print("Average MSE early stop:", avg_mse_earlystop)
        print("Std. dev. MSE early stop:", std_mse_earlystop)
        print("Average final MSE:", avg_mse_fin)
        print("Std. dev. final MSE:", std_mse_fin)
        print("Average final R2:", avg_r2_fin)
        print("Std. dev. final R2:", std_r2_fin)

        all_summary_rows.append({
            "tau_fn": tau_fn,
            "iv_strength": iv_strength,
            "dgp": dgp,
            "num_data": num_data,
            "g_features": g_features,
            "n_centers": n_centers,
            "estimator": estimator,
            "MC_successful_runs": successful_runs,
            "avg_MSEearlystop": avg_mse_earlystop,
            "std_MSEearlystop": std_mse_earlystop,
            "avg_MSEfin": avg_mse_fin,
            "std_MSEfin": std_mse_fin,
            "avg_R2fin": avg_r2_fin,
            "std_R2fin": std_r2_fin
        })

        pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False)

    print("\nExperiment finished.")
    print("Saved:")
    print("-", raw_path)
    print("-", summary_path)


if __name__ == "__main__":
    main()