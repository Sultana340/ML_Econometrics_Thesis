"""
Thesis AGMM vs DeepIV Comparison Script

This script compares baseline AGMM and DeepIV under the same DGP settings.
The comparison is restricted to the z_image design because the current DeepIV
implementation is configured for scalar treatment and high-dimensional instruments.
"""

import os

# Optional: reduce TensorFlow CPU log messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import itertools
import sys
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

# --------------------------------------------------
# Import both experiment functions
# --------------------------------------------------
from run_agmm_experiment import experiment as agmm_experiment
from run_deepiv_experiment import experiment as deepiv_experiment

warnings.simplefilter("ignore", category=UserWarning)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def safe_nanmean(x):
    x = np.array(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.nan
    return np.nanmean(x)


def safe_nanstd(x):
    x = np.array(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.nan
    return np.nanstd(x)


def extract_agmm_metrics(result):
    """
    Extract MSEearlystop and R2fin from AGMM output.

    AGMM output order:
    [R2avg, R2fin, R2earlystop, MSEavg, MSEfin, MSEearlystop]
    """

    if isinstance(result, dict):
        mse = result.get("MSEearlystop", result.get("MSEfin", np.nan))
        r2 = result.get("R2fin", result.get("R2avg", np.nan))
        return mse, r2

    result_array = np.asarray(result, dtype=float).ravel()

    AGMM_R2_INDEX = 1
    AGMM_MSE_INDEX = 5

    mse = result_array[AGMM_MSE_INDEX] if len(result_array) > AGMM_MSE_INDEX else np.nan
    r2 = result_array[AGMM_R2_INDEX] if len(result_array) > AGMM_R2_INDEX else np.nan

    return mse, r2


def extract_deepiv_metrics(result):
    """
    Extract MSEearlystop and R2fin from DeepIV dictionary output.
    """

    mse = result.get("MSEearlystop", np.nan)
    r2 = result.get("R2fin", np.nan)

    return mse, r2


def main():
    device = "cpu"
    print("Running Thesis AGMM vs DeepIV final comparison on", device)

    VERBOSE = False

    # --------------------------------------------------
    # Final CPU-feasible comparison settings
    # --------------------------------------------------
    tau_fn_list = ["abs","sin"]
    iv_strength_list = [0.6]
    dgps = ["z_image"]
    num_data_list = [2000]

    estimators = ["AGMM", "DeepIV"]

    monte_carlo = 5
    n_epochs = 200
    batch_size = 100

    settings = list(
        itertools.product(
            tau_fn_list,
            iv_strength_list,
            dgps,
            num_data_list,
            estimators,
        )
    )

    # --------------------------------------------------
    # Results folder
    # --------------------------------------------------
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    raw_rows = []
    summary_rows = []

    # --------------------------------------------------
    # Main experiment loop
    # --------------------------------------------------
    for tau_fn, iv_strength, dgp, num_data, est in settings:

        print("\n------ Settings ------")
        print("tau_fn:", tau_fn)
        print("iv_strength:", iv_strength)
        print("dgp:", dgp)
        print("num_data:", num_data)
        print("estimator:", est)

        mse_list = []
        r2_list = []

        for run in range(1, monte_carlo + 1):
            print(f"\nRun {run} of {monte_carlo}")

            try:
                if est == "AGMM":
                    # AGMM uses its internal training setup.
                    # Your previous smoke test showed AGMM reaching epoch #199,
                    # so this corresponds to 200 epochs in your current AGMM setup.
                    result = agmm_experiment(
                        dgp,
                        iv_strength,
                        tau_fn,
                        num_data,
                        est,
                        device,
                        VERBOSE,
                    )

                    mse, r2 = extract_agmm_metrics(result)

                elif est == "DeepIV":
                    # DeepIV explicitly receives n_epochs = 200 here.
                    result = deepiv_experiment(
                        dgp=dgp,
                        iv_strength=iv_strength,
                        tau_fn=tau_fn,
                        num_data=num_data,
                        est=est,
                        device=device,
                        DEBUG=VERBOSE,
                        epochs=n_epochs,
                        batch_size=batch_size,
                    )

                    mse, r2 = extract_deepiv_metrics(result)

                else:
                    raise ValueError(f"Unknown estimator: {est}")

            except Exception as e:
                print("\nERROR in setting:")
                print("tau_fn:", tau_fn)
                print("iv_strength:", iv_strength)
                print("dgp:", dgp)
                print("num_data:", num_data)
                print("estimator:", est)
                print("run:", run)
                print("Error message:", e)

                mse = np.nan
                r2 = np.nan

            mse_list.append(mse)
            r2_list.append(r2)

            raw_rows.append(
                {
                    "tau_fn": tau_fn,
                    "iv_strength": iv_strength,
                    "dgp": dgp,
                    "num_data": num_data,
                    "estimator": est,
                    "run": run,
                    "MSEearlystop": mse,
                    "R2fin": r2,
                }
            )

        # --------------------------------------------------
        # Summary statistics
        # --------------------------------------------------
        avg_mse = safe_nanmean(mse_list)
        std_mse = safe_nanstd(mse_list)

        avg_r2 = safe_nanmean(r2_list)
        std_r2 = safe_nanstd(r2_list)

        print("\n---------- Results ----------")
        print("Average MSE early stop:", avg_mse)
        print("Std. dev. MSE early stop:", std_mse)
        print("Average final R2:", avg_r2)
        print("Std. dev. final R2:", std_r2)

        summary_rows.append(
            {
                "tau_fn": tau_fn,
                "iv_strength": iv_strength,
                "dgp": dgp,
                "num_data": num_data,
                "estimator": est,
                "avg_MSEearlystop": avg_mse,
                "std_MSEearlystop": std_mse,
                "avg_R2fin": avg_r2,
                "std_R2fin": std_r2,
            }
        )

    # --------------------------------------------------
    # Save combined raw and summary files
    # --------------------------------------------------
    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(summary_rows)

    raw_path = os.path.join(results_dir, "agmm_vs_deepiv_MC5_raw_final.csv")
    summary_path = os.path.join(results_dir, "agmm_vs_deepiv_MC5_summary_final.csv")

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nExperiment finished.")
    print("Raw results saved to:", raw_path)
    print("Summary results saved to:", summary_path)


if __name__ == "__main__":
    main()