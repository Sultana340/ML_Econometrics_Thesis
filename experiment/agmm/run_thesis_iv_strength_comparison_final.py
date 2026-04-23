"""
 Thesis AGMM Experiment Script
 Derived from modified baseline implementation by Dikkala, Nishanth, Greg Lewis, Lester Mackey, and Vasilis Syrgkanis. "Minimax estimation of conditional moment models." arXiv preprint arXiv:2006.07201 (2020). https://github.com/microsoft/AdversarialGMM.git.
 All modification for thesis experiments by Abida Sultana should be noted in the git history.
 ## Experiment 7: Run Started: 2026-04-21, 11:20 (IV Strength Comparision)
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

    tau_fn_list = ['abs']
    iv_strength_list = [0.3, 0.6, 0.9]
    estimators = ['AGMM']
    dgps = ['z_image']
    num_data_list = [2000]
    monte_carlo = 5

    settings = list(itertools.product(
        tau_fn_list, iv_strength_list, dgps, num_data_list, estimators
    ))

    result_dict = {}
    all_raw_rows = []
    all_summary_rows = []

    os.makedirs("results", exist_ok=True)

    for tau_fn, iv_strength, dgp, num_data, est in settings:
        print("------ Settings ------")
        print("tau_fn:", tau_fn)
        print("iv_strength:", iv_strength)
        print("dgp:", dgp)
        print("num_data:", num_data)
        print("estimator:", est)

        results = []

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

            results.append(mse_earlystop)

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

        np_results = np.array(results)
        avg_mse = np.mean(np_results)
        std_mse = np.std(np_results)

        result_dict[(tau_fn, iv_strength, dgp, num_data, est)] = avg_mse

        print("---------- Results ----------")
        print("Average MSE", avg_mse)
        print("Standard deviation MSE", std_mse)

        all_summary_rows.append({
            'tau_fn': tau_fn,
            'iv_strength': iv_strength,
            'dgp': dgp,
            'num_data': num_data,
            'estimator': est,
            'avg_MSEearlystop': avg_mse,
            'std_MSEearlystop': std_mse
        })

    raw_df = pd.DataFrame(all_raw_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    raw_df.to_csv("results/iv_strength_comparison_raw_final.csv", index=False)
    summary_df.to_csv("results/iv_strength_comparison_summary_final.csv", index=False)

    print("\nSaved:")
    print("- results/iv_strength_comparison_raw_final.csv")
    print("- results/iv_strength_comparison_summary_final.csv")


if __name__ == "__main__":
    main()