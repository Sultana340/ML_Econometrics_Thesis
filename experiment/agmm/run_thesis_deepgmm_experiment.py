import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------
# Project paths
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# ML_Econometrics_Thesis/experiment/agmm
experiment_dir = os.path.dirname(script_dir)

# ML_Econometrics_Thesis
project_root = os.path.dirname(experiment_dir)

# For importing src/agmm
src_path = os.path.join(project_root, "src")

# For importing experiment/agmm/deepgmm/methods, models, learning, etc.
deepgmm_path = os.path.join(script_dir, "deepgmm")

sys.path.insert(0, src_path)
sys.path.insert(0, deepgmm_path)
sys.path.insert(0, script_dir)


# --------------------------------------------------
# Imports
# --------------------------------------------------
from agmm.iv_dgp_generate_data import generate_data
from agmm.agmm_utilities import eval_performance

from methods.mnist_z_model_selection_method import MNISTZModelSelectionMethod

warnings.simplefilter("ignore", category=UserWarning)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def dgp_to_bools(dgp_str):
    """
    Converts DGP name into image-design booleans.

    z_image is the correct design for the current DeepGMM class.
    """
    x_image = False
    z_image = False

    if dgp_str == "x_image":
        x_image = True
    elif dgp_str == "z_image":
        z_image = True
    elif dgp_str == "xz_image":
        x_image = True
        z_image = True
    else:
        raise ValueError(f"Unknown dgp: {dgp_str}")

    return x_image, z_image


def as_tensor(array, device="cpu", dtype=torch.float64):
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)

    return torch.as_tensor(array, device=device, dtype=dtype)


def ensure_2d_column(tensor):
    """
    Ensures scalar variables have shape (n, 1).
    """
    if tensor.dim() == 1:
        return tensor.view(-1, 1)
    return tensor


def ensure_image_tensor(z_tensor):
    """
    Makes image-valued instruments compatible with CNN input.

    Accepts:
        (n, 28, 28)      -> (n, 1, 28, 28)
        (n, 784)         -> (n, 1, 28, 28)
        (n, 1, 28, 28)   -> unchanged
    """
    if z_tensor.dim() == 2 and z_tensor.shape[1] == 784:
        return z_tensor.view(-1, 1, 28, 28)

    if z_tensor.dim() == 3:
        return z_tensor.unsqueeze(1)

    return z_tensor


def clean_result_value(value):
    """
    Converts result values to CSV-safe Python scalars where possible.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return value.tolist()

    if isinstance(value, (np.floating, np.integer)):
        return value.item()

    return value


# --------------------------------------------------
# DeepGMM experiment
# --------------------------------------------------
def experiment(
    dgp,
    iv_strength,
    tau_fn,
    num_data,
    est="DeepGMM",
    device="cpu",
    DEBUG=False,
):
    """
    Runs one DeepGMM experiment and returns the performance dictionary.

    Parameters
    ----------
    dgp : str
        Must be "z_image" for this DeepGMM implementation.
    iv_strength : float
        Instrument strength parameter.
    tau_fn : str
        Structural function name, e.g. "abs", "sin", "linear".
    num_data : int
        Number of training samples.
    est : str
        Kept for compatibility with AGMM-style runners. Must be "DeepGMM".
    device : str
        "cpu" or "cuda".
    DEBUG : bool
        If True, prints additional training output.
    """
    if est != "DeepGMM":
        raise ValueError(
            f"run_thesis_deepgmm_experiment only supports est='DeepGMM'. "
            f"Received est={est}."
        )

    if dgp != "z_image":
        raise ValueError(
            "The current MNISTZModelSelectionMethod is designed only for "
            "dgp='z_image' because it uses an MLP for scalar treatment T/X "
            "and a CNN for image-valued instrument Z."
        )

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # --------------------------------------------------
    # Generate data using the same generator as AGMM
    # --------------------------------------------------
    X_IMAGE, Z_IMAGE = dgp_to_bools(dgp)
    n_instruments = 1

    data = generate_data(
        X_IMAGE=X_IMAGE,
        Z_IMAGE=Z_IMAGE,
        tau_fn=tau_fn,
        n_samples=num_data,
        n_dev_samples=num_data // 2,
        n_instruments=n_instruments,
        iv_strength=iv_strength,
        device=device,
    )

    Z_train, T_train, Y_train, G_train = data[0]
    Z_val, T_val, Y_val, G_val = data[1]
    Z_test, T_test, Y_test, G_test = data[2]
    Z_dev, T_dev, Y_dev, G_dev = data[3]

    # --------------------------------------------------
    # Tensor formatting for DeepGMM
    # --------------------------------------------------
    # The DeepGMM structural learner g is double precision MLP.
    T_train = ensure_2d_column(as_tensor(T_train, device=device, dtype=torch.float64))
    Y_train = ensure_2d_column(as_tensor(Y_train, device=device, dtype=torch.float64))

    T_dev = ensure_2d_column(as_tensor(T_dev, device=device, dtype=torch.float64))
    Y_dev = ensure_2d_column(as_tensor(Y_dev, device=device, dtype=torch.float64))
    G_dev = ensure_2d_column(as_tensor(G_dev, device=device, dtype=torch.float64))

    T_test = ensure_2d_column(as_tensor(T_test, device=device, dtype=torch.float64))
    G_test = ensure_2d_column(as_tensor(G_test, device=device, dtype=torch.float64))

    # The DeepGMM adversary/test function f is a CNN. Most CNN modules are float32.
    # If your DefaultCNN is explicitly double precision, change dtype here to torch.float64.
    Z_train = ensure_image_tensor(as_tensor(Z_train, device=device, dtype=torch.float32))
    Z_dev = ensure_image_tensor(as_tensor(Z_dev, device=device, dtype=torch.float32))

    # --------------------------------------------------
    # Train DeepGMM
    # --------------------------------------------------
    enable_cuda = device == "cuda"

    estimator = MNISTZModelSelectionMethod(enable_cuda=enable_cuda)

    estimator.fit(
        x_train=T_train,
        z_train=Z_train,
        y_train=Y_train,
        x_dev=T_dev,
        z_dev=Z_dev,
        y_dev=Y_dev,
        verbose=DEBUG,
        g_dev=G_dev,
    )

    # --------------------------------------------------
    # Evaluate structural recovery
    # --------------------------------------------------
    results = eval_performance(
        estimator,
        T_test,
        true_of_T_test=G_test,
    )

    return results


# --------------------------------------------------
# Thesis grid runner
# --------------------------------------------------
def summarize_results(raw_df):
    """
    Creates a summary table with average and standard deviation of all numeric
    metric columns across Monte Carlo replications.
    """
    id_cols = ["estimator", "dgp", "tau_fn", "iv_strength", "num_data"]
    exclude_cols = set(id_cols + ["run"])

    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [col for col in numeric_cols if col not in exclude_cols]

    if not metric_cols:
        return pd.DataFrame()

    summary = (
        raw_df
        .groupby(id_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten MultiIndex columns.
    flat_cols = []
    for col in summary.columns:
        if isinstance(col, tuple):
            name, stat = col
            if stat == "":
                flat_cols.append(name)
            elif stat == "mean":
                flat_cols.append(f"avg_{name}")
            elif stat == "std":
                flat_cols.append(f"std_{name}")
            else:
                flat_cols.append(f"{stat}_{name}")
        else:
            flat_cols.append(col)

    summary.columns = flat_cols
    return summary


def main():
    device = "cpu"
    DEBUG = False

    print("Running Thesis DeepGMM experiment on", device)

    # --------------------------------------------------
    # Thesis settings
    # --------------------------------------------------
    # Start with monte_carlo = 1 for a smoke test.
    # For final thesis results, use monte_carlo = 3 or 5 if CPU time allows.
    tau_fn_list = ["abs", "sin"]
    iv_strength_list = [0.6]
    dgps = ["z_image"]
    num_data_list = [2000]
    monte_carlo = 5

    settings = list(itertools.product(
        tau_fn_list,
        iv_strength_list,
        dgps,
        num_data_list,
    ))

    print(f"Total settings: {len(settings)}")
    print(f"Monte Carlo replications per setting: {monte_carlo}")

    raw_rows = []

    for tau_fn, iv_strength, dgp, num_data in settings:
        print("\n---------- Settings ----------")
        print("estimator:", "DeepGMM")
        print("tau_fn:", tau_fn)
        print("iv_strength:", iv_strength)
        print("dgp:", dgp)
        print("num_data:", num_data)

        for run in range(1, monte_carlo + 1):
            print(f"\nRun {run} of {monte_carlo}")

            row = {
                "estimator": "DeepGMM",
                "tau_fn": tau_fn,
                "iv_strength": iv_strength,
                "dgp": dgp,
                "num_data": num_data,
                "run": run,
            }

            try:
                results = experiment(
                    dgp=dgp,
                    iv_strength=iv_strength,
                    tau_fn=tau_fn,
                    num_data=num_data,
                    est="DeepGMM",
                    device=device,
                    DEBUG=DEBUG,
                )

                cleaned_results = {
                    key: clean_result_value(value)
                    for key, value in results.items()
                }

                row.update(cleaned_results)

                print("Result:", cleaned_results)

            except Exception as exc:
                print("\nERROR in setting:")
                print("tau_fn:", tau_fn)
                print("iv_strength:", iv_strength)
                print("dgp:", dgp)
                print("num_data:", num_data)
                print("run:", run)
                print("Error message:", exc)

                row["error"] = str(exc)

            raw_rows.append(row)

    # --------------------------------------------------
# Save results
# --------------------------------------------------
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

raw_df = pd.DataFrame(raw_rows)
raw_path = os.path.join(results_dir, "agmm_deepgmm_raw.csv")
raw_df.to_csv(raw_path, index=False)

summary_df = summarize_results(raw_df)
summary_path = os.path.join(results_dir, "agmm_deepgmm_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n---------- Finished ----------")
print("Raw results saved to:", raw_path)
print("Summary results saved to:", summary_path)

if not summary_df.empty:
    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()
