"""
 Thesis AGMM Experiment Script
 Derived from modified baseline implementation by Dikkala, Nishanth, Greg Lewis, Lester Mackey, and Vasilis Syrgkanis. "Minimax estimation of conditional moment models." arXiv preprint arXiv:2006.07201 (2020). https://github.com/microsoft/AdversarialGMM.git.
 All modification for thesis experiments by Abida Sultana should be noted in the git history.
 Objective of this script is to compare the performance of different tau functions (linear, sin, abs) in the AGMM framework for a specific DGP (z_image) and IV strength (0.5) with a fixed sample size (2000).
  The results will be saved in CSV files for further analysis and visualization.
"""


import itertools # For creating the Cartesian product of settings
import os # For file path manipulations
import warnings
import numpy as np
import pandas as pd

# Add the src directory to the path to import custom modules
import sys # 
# Get the project root directory (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
project_root = os.path.dirname(os.path.dirname(script_dir)) # Get the parent directory of the script directory, which is the project root
src_path = os.path.join(project_root, 'src') # Construct the path to the src directory
sys.path.insert(0, src_path) # Add the src directory to the system path for imports
sys.path.insert(0, script_dir)# Add the current script directory to the system path for imports

from run_agmm_experiment import experiment 
warnings.simplefilter("ignore", category=UserWarning) 

def main():
 device = 'cpu'
 print("Running Thesis AGMM experiment on", device) 
 VERBOSE = False

 tau_fn = ['linear','sin','abs']
 iv_strength = [0.5]
 estimators = ['AGMM'] 
 dgps = ['x_image']
 num_datas = [2000]

 settings = list(itertools.product(
    tau_fn, iv_strength, dgps, num_datas, estimators))
 result_dict = {}
 monte_carlo = 5

 for( tau_fn, iv_strength, dgp, num_data, est) in settings:
    print("------ Settings ------")
    print(tau_fn)
    print("iv_strength", iv_strength)
    print("dgp", dgp)
    print("estimator", est)
    results = []
    for run in range(monte_carlo):
        print("Run", run+1)
        result= experiment(dgp, iv_strength, tau_fn, 
                           num_data, est, device, VERBOSE)
        results.append(list(result))

    np_results = np.array(results)
    result_dict[(tau_fn, iv_strength, dgp, num_data, est)] = np_results.mean(axis=0)
    print("---------- Results ----------")
    print("Average MSE", np_results.mean(axis=0)[5])
    print("Standard deviation MSE", np_results.std(axis=0)[5])

    os.makedirs("../../Analysis/tau_comparison", exist_ok=True)

    avg_mse = np_results.mean(axis=0)[5]
    std_mse = np_results.std(axis=0)[5]

    # Raw results
    df = pd.DataFrame({
        'tau_fn': [tau_fn] * monte_carlo,
        'iv_strength': [iv_strength] * monte_carlo,
        'dgp': [dgp] * monte_carlo,
        'num_data': [num_data] * monte_carlo,
        'estimator': [est] * monte_carlo,
        'MSE': np_results[:, 5]
    })
    df.to_csv(f"../../Analysis/tau_comparison/raw_results_{tau_fn}_{iv_strength}_{dgp}_{num_data}_{est}.csv", index=False)
    # Summary results
    summary_df = pd.DataFrame({
        'tau_fn': [tau_fn],
        'iv_strength': [iv_strength],
        'dgp': [dgp],
        'num_data': [num_data],
        'estimator': [est],
        'avg_MSE': [avg_mse],
        'std_MSE': [std_mse]
    })
    summary_df.to_csv(f"../../Analysis/tau_comparison/summary_results_{tau_fn}_{iv_strength}_{dgp}_{num_data}_{est}.csv", index=False)
if __name__ == "__main__":
    main()