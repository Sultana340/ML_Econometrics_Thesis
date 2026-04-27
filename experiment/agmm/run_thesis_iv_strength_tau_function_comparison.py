"""
 Thesis AGMM Experiment Script
 Derived from modified baseline implementation by Dikkala, Nishanth, Greg Lewis, Lester Mackey, and Vasilis Syrgkanis. "Minimax estimation of conditional moment models." arXiv preprint arXiv:2006.07201 (2020). https://github.com/microsoft/AdversarialGMM.git.
 All modification for thesis experiments by Abida Sultana should be noted in the git history.
## Experiment 8: How effectively does AGMM recover the structural function
under varying levels of identification and structural complexity?
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

# add src and script dir to sys.path for imports
sys.path.insert(0, src_path)
sys.path.insert(0, script_dir)

from run_agmm_experiment import experiment
warnings.simplefilter("ignore", category=UserWarning)


def main():
    device = 'cpu'
    print("Running Final IV Strength Comparison Thesis AGMM experiment on", device)
    VERBOSE = False

    tau_fn_list = ['linear']   # add 'linear' if supported
    iv_strength_list = [0.3, 0.6, 0.9]
    estimators = ['AGMM']
    dgps = ['z_image']
    num_data_list = [2000]
    monte_carlo = 5

    settings = list(itertools.product(
        tau_fn_list, iv_strength_list, dgps, num_data_list, estimators
    ))

    all_raw_rows = []
    all_summary_rows = []

    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    for tau_fn, iv_strength, dgp, num_data, est in settings:
        print("------ Settings ------")
        print("tau_fn:", tau_fn)
        print("iv_strength:", iv_strength)
        print("dgp:", dgp)
        print("num_data:", num_data)
        print("estimator:", est)

        mse_results = []
        r2_results = []

        for run in range(monte_carlo):
            print("Run", run + 1)

            result = experiment(
                dgp, iv_strength, tau_fn, num_data, est, device, VERBOSE
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
                'estimator': est,
                'run': run + 1,
                'R2_avg': r2_avg,
                'R2_fin': r2_fin,
                'MSE_avg': mse_avg,
                'MSE_fin': mse_fin,
                'MSE_earlystop': mse_earlystop
            })

        avg_mse = np.mean(mse_results)
        std_mse = np.std(mse_results, ddof=1)
        avg_r2 = np.mean(r2_results)
        std_r2 = np.std(r2_results, ddof=1)

        print("---------- Results ----------")
        print("Average MSE (early stop):", avg_mse)
        print("Std. dev. MSE (early stop):", std_mse)
        print("Average final R2:", avg_r2)
        print("Std. dev. final R2:", std_r2)

        all_summary_rows.append({
            'tau_fn': tau_fn,
            'iv_strength': iv_strength,
            'dgp': dgp,
            'num_data': num_data,
            'estimator': est,
            'avg_MSEearlystop': avg_mse,
            'std_MSEearlystop': std_mse,
            'avg_R2fin': avg_r2,
            'std_R2fin': std_r2
        })

    raw_df = pd.DataFrame(all_raw_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    raw_df.to_csv(
        os.path.join(results_dir, "iv_strength_tau_function_comparison_raw_final.csv"),
        index=False
    )
    summary_df.to_csv(
        os.path.join(results_dir, "iv_strength_tau_function_comparison_summary_final.csv"),
        index=False
    )

    print("\nSaved:")
    print("-", os.path.join(results_dir, "iv_strength_tau_function_comparison_raw_final.csv"))
    print("-", os.path.join(results_dir, "iv_strength_tau_function_comparison_summary_final.csv"))


if __name__ == "__main__":
    main()