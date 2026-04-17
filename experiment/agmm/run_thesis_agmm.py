import itertools
import os
import numpy as np
import pandas as pd

# Add the src directory to the path to import custom modules
import sys
# Get the project root directory (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, script_dir)

from run_agmm_experiment import experiment 


def main():
 device = 'cpu'
 print("Running Thesis AGMM experiment on", device) # 
 VERBOSE = False

 tau_fn = ['abs']
 iv_strength = [0.5]
 estimators = ['AGMM']
 dgps = ['z_image']
 num_datas = [1000]

 settings = list(itertools.product(
    tau_fn, iv_strength, dgps, num_datas, estimators))
 result_dict = {}
 monte_carlo = 1

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

    os.makedirs("results", exist_ok=True)

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
    df.to_csv(f"results/raw_results_{tau_fn}_{iv_strength}_{dgp}_{num_data}_{est}.csv", index=False)
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
    summary_df.to_csv(f"results/summary_results_{tau_fn}_{iv_strength}_{dgp}_{num_data}_{est}.csv", index=False)
if __name__ == "__main__":
    main()
    