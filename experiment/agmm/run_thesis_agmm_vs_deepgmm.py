import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------
# Project paths based on your current structure
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# script_dir:
# ML_Econometrics_Thesis/experiment/agmm
experiment_dir = os.path.dirname(script_dir)

# project_root:
# ML_Econometrics_Thesis
project_root = os.path.dirname(experiment_dir)

# AGMM package is in src/agmm
src_path = os.path.join(project_root, "src")

# DeepGMM folders are in experiment/agmm/deepgmm
deepgmm_path = os.path.join(script_dir, "deepgmm")

sys.path.insert(0, src_path)
sys.path.insert(0, deepgmm_path)
sys.path.insert(0, script_dir)


# --------------------------------------------------
# Imports
# --------------------------------------------------
from agmm.iv_dgp_generate_data import generate_data
from agmm.agmm_trainer import train_agmm
from agmm.agmm_utilities import eval_performance

from methods.mnist_z_model_selection_method import MNISTZModelSelectionMethod


warnings.simplefilter("ignore", category=UserWarning)


# --------------------------------------------------
# General helpers
# --------------------------------------------------
def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dgp_to_bools(dgp_str):
    """
    Convert DGP string into X_IMAGE and Z_IMAGE flags.
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
    """
    Convert numpy array or torch tensor to torch tensor.
    """
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)

    return torch.as_tensor(array, device=device, dtype=dtype)


def ensure_2d_column(tensor):
    """
    Ensure scalar variables have shape (n, 1).
    """
    if tensor.dim() == 1:
        return tensor.view(-1, 1)

    return tensor


def ensure_image_tensor(z_tensor):
    """
    Ensure MNIST image instruments have CNN-compatible shape.

    Possible input shapes:
        (n, 784)        -> (n, 1, 28, 28)
        (n, 28, 28)     -> (n, 1, 28, 28)
        (n, 1, 28, 28)  -> unchanged
    """
    if z_tensor.dim() == 2 and z_tensor.shape[1] == 784:
        return z_tensor.view(-1, 1, 28, 28)

    if z_tensor.dim() == 3:
        return z_tensor.unsqueeze(1)

    return z_tensor


def clean_result_value(value):
    """
    Convert result values to CSV-safe Python scalars where possible.
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


def normalize_results(results):
    """
    Convert performance output into a dictionary.

    In this project, AGMM eval_performance may return either:
        - a dictionary
        - a tuple/list in the order:
          R2avg, R2fin, R2earlystop, MSEavg, MSEfin, MSEearlystop

    DeepGMM evaluation already returns a dictionary, but this function keeps
    the output handling uniform.
    """
    if isinstance(results, dict):
        return {
            key: clean_result_value(value)
            for key, value in results.items()
        }

    if isinstance(results, (tuple, list)):
        metric_names = [
            "R2avg",
            "R2fin",
            "R2earlystop",
            "MSEavg",
            "MSEfin",
            "MSEearlystop",
        ]

        if len(results) == len(metric_names):
            return {
                metric_names[i]: clean_result_value(results[i])
                for i in range(len(metric_names))
            }

        return {
            f"metric_{i}": clean_result_value(value)
            for i, value in enumerate(results)
        }

    return {"result": clean_result_value(results)}


def eval_deepgmm_performance(estimator, T_test, G_test):
    """
    Evaluate DeepGMM structural recovery.

    DeepGMM does not have AGMM-style averaged and early-stop prediction APIs.
    Therefore, for DeepGMM we evaluate the final selected structural function
    and report the common comparison metrics:

        R2fin
        MSEfin

    These are directly comparable with AGMM's R2fin and MSEfin.
    """
    if estimator.g is None:
        raise AttributeError("DeepGMM estimator has no fitted structural model g.")

    estimator.g.eval()

    with torch.no_grad():
        pred = estimator.predict(T_test)

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().double()
    else:
        pred = torch.as_tensor(pred, dtype=torch.float64)

    if isinstance(G_test, torch.Tensor):
        true = G_test.detach().cpu().double()
    else:
        true = torch.as_tensor(G_test, dtype=torch.float64)

    pred = pred.view(-1)
    true = true.view(-1)

    mse = torch.mean((true - pred) ** 2).item()
    var_true = torch.var(true, unbiased=False).item()

    r2 = np.nan
    if var_true > 0:
        r2 = 1.0 - mse / var_true

    return {
        "R2fin": r2,
        "MSEfin": mse,
    }


# --------------------------------------------------
# AGMM hyperparameters
# --------------------------------------------------
def agmm_hyperparams(dgp):
    """
    AGMM hyperparameters matched to your run_agmm_experiment.py for AGMM.

    For dgp='z_image', your AGMM script uses learner_lr = 2e-4.
    """
    dropout_p = 0.1
    n_hidden = 200

    learner_lr = 1e-4
    adversary_lr = 5e-5
    learner_l2 = 1e-4
    adversary_l2 = 1e-4
    adversary_norm_reg = 1e-4
    n_epochs = 200
    batch_size = 100
    train_learner_every = 1
    train_adversary_every = 2

    if dgp == "z_image":
        learner_lr = 2e-4
    elif dgp == "x_image":
        learner_l2 = 1e-8
        adversary_l2 = 1e-8

    return {
        "dropout_p": dropout_p,
        "n_hidden": n_hidden,
        "learner_lr": learner_lr,
        "adversary_lr": adversary_lr,
        "learner_l2": learner_l2,
        "adversary_l2": adversary_l2,
        "adversary_norm_reg": adversary_norm_reg,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "train_learner_every": train_learner_every,
        "train_adversary_every": train_adversary_every,
    }


# --------------------------------------------------
# Train/evaluate AGMM on already-generated data
# --------------------------------------------------
def run_agmm_on_data(data, dgp, device="cpu", DEBUG=False):
    """
    Train AGMM on a fixed generated data split and return performance metrics.
    """
    X_IMAGE, Z_IMAGE = dgp_to_bools(dgp)

    Z_train, T_train, Y_train, G_train = data[0]
    Z_val, T_val, Y_val, G_val = data[1]
    Z_test, T_test, Y_test, G_test = data[2]
    Z_dev, T_dev, Y_dev, G_dev = data[3]

    hp = agmm_hyperparams(dgp)

    estimator = train_agmm(
        Z_train, T_train, Y_train, G_train,
        Z_dev, T_dev, Y_dev, G_dev,
        Z_val, T_val, Y_val, G_val,
        T_test, G_test,
        X_IMAGE=X_IMAGE,
        Z_IMAGE=Z_IMAGE,
        n_t=1,
        n_instruments=1,
        n_hidden=hp["n_hidden"],
        dropout_p=hp["dropout_p"],
        learner_lr=hp["learner_lr"],
        adversary_lr=hp["adversary_lr"],
        learner_l2=hp["learner_l2"],
        adversary_l2=hp["adversary_l2"],
        adversary_norm_reg=hp["adversary_norm_reg"],
        n_epochs=hp["n_epochs"],
        batch_size=hp["batch_size"],
        train_learner_every=hp["train_learner_every"],
        train_adversary_every=hp["train_adversary_every"],
        device=device,
        DEBUG=DEBUG,
    )

    results = eval_performance(
        estimator,
        T_test,
        true_of_T_test=G_test,
    )

    return results


# --------------------------------------------------
# Train/evaluate DeepGMM on already-generated data
# --------------------------------------------------
def run_deepgmm_on_data(data, dgp, device="cpu", DEBUG=False):
    """
    Train DeepGMM on a fixed generated data split and return performance metrics.
    """
    if dgp != "z_image":
        raise ValueError(
            "Current DeepGMM implementation supports only dgp='z_image'. "
            "It uses an MLP for scalar treatment T/X and a CNN for image-valued Z."
        )

    Z_train, T_train, Y_train, G_train = data[0]
    Z_val, T_val, Y_val, G_val = data[1]
    Z_test, T_test, Y_test, G_test = data[2]
    Z_dev, T_dev, Y_dev, G_dev = data[3]

    # DeepGMM structural learner g is double precision MLP.
    T_train_d = ensure_2d_column(as_tensor(T_train, device=device, dtype=torch.float64))
    Y_train_d = ensure_2d_column(as_tensor(Y_train, device=device, dtype=torch.float64))

    T_dev_d = ensure_2d_column(as_tensor(T_dev, device=device, dtype=torch.float64))
    Y_dev_d = ensure_2d_column(as_tensor(Y_dev, device=device, dtype=torch.float64))
    G_dev_d = ensure_2d_column(as_tensor(G_dev, device=device, dtype=torch.float64))

    T_test_d = ensure_2d_column(as_tensor(T_test, device=device, dtype=torch.float64))
    G_test_d = ensure_2d_column(as_tensor(G_test, device=device, dtype=torch.float64))

    # DeepGMM adversary/test function f is CNN over Z images.
    # Your DefaultCNN uses double precision parameters, so Z must also be double.
    Z_train_f = ensure_image_tensor(as_tensor(Z_train, device=device, dtype=torch.float64))
    Z_dev_f = ensure_image_tensor(as_tensor(Z_dev, device=device, dtype=torch.float64))

    enable_cuda = device == "cuda"

    estimator = MNISTZModelSelectionMethod(enable_cuda=enable_cuda)

    estimator.fit(
        x_train=T_train_d,
        z_train=Z_train_f,
        y_train=Y_train_d,
        x_dev=T_dev_d,
        z_dev=Z_dev_f,
        y_dev=Y_dev_d,
        verbose=DEBUG,
        g_dev=G_dev_d,
    )

    # Do not use AGMM's eval_performance here.
    # It passes AGMM-specific keyword arguments such as burn_in to predict().
    results = eval_deepgmm_performance(
        estimator=estimator,
        T_test=T_test_d,
        G_test=G_test_d,
    )

    return results


# --------------------------------------------------
# One paired comparison
# --------------------------------------------------
def run_paired_comparison(
    dgp,
    iv_strength,
    tau_fn,
    num_data,
    run,
    base_seed=12345,
    device="cpu",
    DEBUG=False,
):
    """
    Generate one dataset and run both AGMM and DeepGMM on it.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"

    if dgp != "z_image":
        raise ValueError(
            "This comparison script is intended for dgp='z_image', because "
            "the current DeepGMM method is MNISTZModelSelectionMethod."
        )

    # Unique but reproducible data seed per setting/run.
    tau_code = sum(ord(ch) for ch in str(tau_fn))
    data_seed = int(
        base_seed
        + 100000 * run
        + 1000 * int(round(iv_strength * 10))
        + 17 * tau_code
        + int(num_data)
    )

    X_IMAGE, Z_IMAGE = dgp_to_bools(dgp)

    # Generate the data once.
    set_seed(data_seed)
    data = generate_data(
        X_IMAGE=X_IMAGE,
        Z_IMAGE=Z_IMAGE,
        tau_fn=tau_fn,
        n_samples=num_data,
        n_dev_samples=num_data // 2,
        n_instruments=1,
        iv_strength=iv_strength,
        device=device,
    )

    rows = []

    # --------------------------------------------------
    # AGMM
    # --------------------------------------------------
    agmm_seed = data_seed + 101
    set_seed(agmm_seed)

    row = {
        "estimator": "AGMM",
        "tau_fn": tau_fn,
        "iv_strength": iv_strength,
        "dgp": dgp,
        "num_data": num_data,
        "run": run,
        "data_seed": data_seed,
        "train_seed": agmm_seed,
    }

    start_time = time.time()

    try:
        results = run_agmm_on_data(
            data=data,
            dgp=dgp,
            device=device,
            DEBUG=DEBUG,
        )

        cleaned_results = normalize_results(results)

        row.update(cleaned_results)
        row["elapsed_sec"] = time.time() - start_time
        print("AGMM result:", cleaned_results)

    except Exception as exc:
        row["error"] = str(exc)
        row["elapsed_sec"] = time.time() - start_time
        print("AGMM ERROR:", exc)

    rows.append(row)

    # --------------------------------------------------
    # DeepGMM
    # --------------------------------------------------
    deepgmm_seed = data_seed + 202
    set_seed(deepgmm_seed)

    row = {
        "estimator": "DeepGMM",
        "tau_fn": tau_fn,
        "iv_strength": iv_strength,
        "dgp": dgp,
        "num_data": num_data,
        "run": run,
        "data_seed": data_seed,
        "train_seed": deepgmm_seed,
    }

    start_time = time.time()

    try:
        results = run_deepgmm_on_data(
            data=data,
            dgp=dgp,
            device=device,
            DEBUG=DEBUG,
        )

        cleaned_results = normalize_results(results)

        row.update(cleaned_results)
        row["elapsed_sec"] = time.time() - start_time
        print("DeepGMM result:", cleaned_results)

    except Exception as exc:
        row["error"] = str(exc)
        row["elapsed_sec"] = time.time() - start_time
        print("DeepGMM ERROR:", exc)

    rows.append(row)

    return rows


# --------------------------------------------------
# Summarize Monte Carlo results
# --------------------------------------------------
def summarize_results(raw_df):
    """
    Summarize numerical metric columns across Monte Carlo replications.
    """
    id_cols = ["estimator", "dgp", "tau_fn", "iv_strength", "num_data"]
    exclude_cols = set(id_cols + ["run", "data_seed", "train_seed"])

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


# --------------------------------------------------
# Main grid runner
# --------------------------------------------------
def main():
    device = "cpu"
    DEBUG = False

    print("Running Thesis AGMM vs DeepGMM comparison on", device)
    print("script_dir:", script_dir)
    print("project_root:", project_root)
    print("src_path:", src_path)
    print("deepgmm_path:", deepgmm_path)

    # --------------------------------------------------
    # Use smoke test first.
    # After it works, set SMOKE_TEST = False.
    # --------------------------------------------------
    SMOKE_TEST = True

    if SMOKE_TEST:
        tau_fn_list = ["abs"]
        iv_strength_list = [0.6]
        dgps = ["z_image"]
        num_data_list = [500]
        monte_carlo = 1
    else:
        # Thesis comparison setting.
        tau_fn_list = ["abs", "sin"]
        iv_strength_list = [0.3, 0.6, 0.9]
        dgps = ["z_image"]
        num_data_list = [2000]
        monte_carlo = 3

    settings = list(itertools.product(
        tau_fn_list,
        iv_strength_list,
        dgps,
        num_data_list,
    ))

    print("Total settings:", len(settings))
    print("Monte Carlo replications per setting:", monte_carlo)
    print("SMOKE_TEST:", SMOKE_TEST)

    raw_rows = []

    for tau_fn, iv_strength, dgp, num_data in settings:
        print("\n========================================")
        print("Setting")
        print("tau_fn:", tau_fn)
        print("iv_strength:", iv_strength)
        print("dgp:", dgp)
        print("num_data:", num_data)
        print("========================================")

        for run in range(1, monte_carlo + 1):
            print(f"\nPaired run {run} of {monte_carlo}")

            rows = run_paired_comparison(
                dgp=dgp,
                iv_strength=iv_strength,
                tau_fn=tau_fn,
                num_data=num_data,
                run=run,
                base_seed=12345,
                device=device,
                DEBUG=DEBUG,
            )

            raw_rows.extend(rows)

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    raw_df = pd.DataFrame(raw_rows)
    raw_path = os.path.join(results_dir, "thesis_agmm_deepgmm_comparison_raw.csv")
    raw_df.to_csv(raw_path, index=False)

    summary_df = summarize_results(raw_df)
    summary_path = os.path.join(
        results_dir,
        "thesis_agmm_deepgmm_comparison_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False)

    print("\n---------- Finished ----------")
    print("Raw results saved to:", raw_path)
    print("Summary results saved to:", summary_path)

    if not summary_df.empty:
        print("\nSummary:")
        print(summary_df)


if __name__ == "__main__":
    main()
