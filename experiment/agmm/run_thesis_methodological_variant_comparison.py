"""
 Thesis AGMM Experiment Script
 Derived from modified baseline implementation by Dikkala, Nishanth, Greg Lewis, Lester Mackey, and Vasilis Syrgkanis. "Minimax estimation of conditional moment models." arXiv preprint arXiv:2006.07201 (2020). https://github.com/microsoft/AdversarialGMM.git.
 All modification for thesis experiments by Abida Sultana should be noted in the git history.
## Experiment 9:  Methodological Variant Comparison
"""

import itertools
import os
import warnings
import numpy as np
import pandas as pd
import sys

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_path = os.path.join(project_root, 'src')

# Add src and script dir to sys.path for imports
sys.path.insert(0, src_path)
sys.path.insert(0, script_dir)

from run_agmm_experiment import experiment

warnings.simplefilter("ignore", category=UserWarning)


def main():
    device = 'cpu'
    print("Running Thesis Variant Comparison Experiment on", device)

    VERBOSE = False

    # Main thesis setting for variant comparison
    tau_fn_list = ['abs']
    iv_strength_list = [0.6]
    dgps = ['z_image']
    num_data_list = [2000]

    # Added g_features
    g_features_list = [10]

    monte_carlo = 1

    estimators = ['CentroidMMDGMM']


    settings = list(itertools.product(
        tau_fn_list,
        iv_strength_list,
        dgps,
        num_data_list,
        g_features_list,
        estimators
    ))

    all_raw_rows = []
    all_summary_rows = []

    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    raw_path = os.path.join(
        results_dir,
        "CentroidMMDGMM_raw_abs_pi06_zimage.csv"
    )

    summary_path = os.path.join(
        results_dir,
        "CentroidMMDGMM_summary_abs_pi06_zimage.csv"
    )

    for tau_fn, iv_strength, dgp, num_data, g_features, est in settings:
        print("\n------ Settings ------")
        print("tau_fn:", tau_fn)
        print("iv_strength:", iv_strength)
        print("dgp:", dgp)
        print("num_data:", num_data)
        print("g_features:", g_features)
        print("estimator:", est)

        mse_results = []
        r2_results = []

        for run in range(monte_carlo):
            print("\nRun", run + 1, "of", monte_carlo)

            try:
                result = experiment(
                    dgp,
                    iv_strength,
                    tau_fn,
                    num_data,
                    est,
                    device=device,
                    DEBUG=VERBOSE,
                    g_features=g_features
                )

                r2_avg = float(result[0])
                r2_fin = float(result[1])
                mse_avg = float(result[3])
                mse_fin = float(result[4])
                mse_earlystop = float(result[5])

                mse_results.append(mse_earlystop)
                r2_results.append(r2_fin)

                all_raw_rows.append({
                    'tau_fn': tau_fn,
                    'iv_strength': iv_strength,
                    'dgp': dgp,
                    'num_data': num_data,
                    'g_features': g_features,
                    'estimator': est,
                    'run': run + 1,
                    'R2_avg': r2_avg,
                    'R2_fin': r2_fin,
                    'MSE_avg': mse_avg,
                    'MSE_fin': mse_fin,
                    'MSE_earlystop': mse_earlystop
                })

                # Save raw results after every successful run
                raw_df = pd.DataFrame(all_raw_rows)
                raw_df.to_csv(raw_path, index=False)

                print("Saved partial raw results.")

            except Exception as e:
                print("\nERROR in setting:")
                print("tau_fn:", tau_fn)
                print("iv_strength:", iv_strength)
                print("dgp:", dgp)
                print("num_data:", num_data)
                print("g_features:", g_features)
                print("estimator:", est)
                print("run:", run + 1)
                print("Error message:", e)

                all_raw_rows.append({
                    'tau_fn': tau_fn,
                    'iv_strength': iv_strength,
                    'dgp': dgp,
                    'num_data': num_data,
                    'g_features': g_features,
                    'estimator': est,
                    'run': run + 1,
                    'R2_avg': np.nan,
                    'R2_fin': np.nan,
                    'MSE_avg': np.nan,
                    'MSE_fin': np.nan,
                    'MSE_earlystop': np.nan,
                    'error': str(e)
                })

                raw_df = pd.DataFrame(all_raw_rows)
                raw_df.to_csv(raw_path, index=False)

                continue

        if len(mse_results) > 0:
            avg_mse = np.mean(mse_results)

            if len(mse_results) > 1:
                std_mse = np.std(mse_results, ddof=1)
            else:
                std_mse = 0.0

            avg_r2 = np.mean(r2_results)

            if len(r2_results) > 1:
                std_r2 = np.std(r2_results, ddof=1)
            else:
                std_r2 = 0.0

        else:
            avg_mse = np.nan
            std_mse = np.nan
            avg_r2 = np.nan
            std_r2 = np.nan

        print("\n---------- Results ----------")
        print("Average MSE early stop:", avg_mse)
        print("Std. dev. MSE early stop:", std_mse)
        print("Average final R2:", avg_r2)
        print("Std. dev. final R2:", std_r2)

        all_summary_rows.append({
            'tau_fn': tau_fn,
            'iv_strength': iv_strength,
            'dgp': dgp,
            'num_data': num_data,
            'g_features': g_features,
            'estimator': est,
            'MC_successful_runs': len(mse_results),
            'avg_MSEearlystop': avg_mse,
            'std_MSEearlystop': std_mse,
            'avg_R2fin': avg_r2,
            'std_R2fin': std_r2
        })

        # Save summary after every setting
        summary_df = pd.DataFrame(all_summary_rows)
        summary_df.to_csv(summary_path, index=False)

    print("\nExperiment finished.")
    print("Saved:")
    print("-", raw_path)
    print("-", summary_path)


if __name__ == "__main__":
    main()