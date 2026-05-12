import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------
# Project paths
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# ML_Econometrics_Thesis/experiment
experiment_dir = os.path.dirname(script_dir)

# ML_Econometrics_Thesis
project_root = os.path.dirname(experiment_dir)

# ML_Econometrics_Thesis/src
src_path = os.path.join(project_root, "src")

# ML_Econometrics_Thesis/experiment/agmm/deepgmm
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
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dgp_to_bools(dgp_str):
    if dgp_str == "x_image":
        return True, False

    if dgp_str == "z_image":
        return False, True

    if dgp_str == "xz_image":
        return True, True

    raise ValueError(f"Unknown dgp: {dgp_str}")


def as_tensor(array, device="cpu", dtype=torch.float64):
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)

    return torch.as_tensor(array, device=device, dtype=dtype)


def ensure_2d_column(tensor):
    if tensor.dim() == 1:
        return tensor.view(-1, 1)

    return tensor


def ensure_image_tensor(z_tensor):
    """
    Convert image-valued instruments into CNN-compatible shape.

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
    Convert AGMM or DeepGMM output into a dictionary.

    AGMM eval_performance may return either:
        - dict
        - tuple/list:
          R2avg, R2fin, R2earlystop, MSEavg, MSEfin, MSEearlystop
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


# --------------------------------------------------
# DeepGMM evaluation
# --------------------------------------------------
def eval_deepgmm_performance(estimator, T_test, G_test):
    """
    Evaluate DeepGMM final structural recovery.

    DeepGMM does not provide AGMM-style averaged or early-stop prediction.
    Therefore, only final MSE and final R2 are reported.
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
    AGMM hyperparameters matched to thesis experiment setting.
    """
    hp = {
        "dropout_p": 0.1,
        "n_hidden": 200,
        "learner_lr": 1e-4,
        "adversary_lr": 5e-5,
        "learner_l2": 1e-4,
        "adversary_l2": 1e-4,
        "adversary_norm_reg": 1e-4,
        "n_epochs": 200,
        "batch_size": 100,
        "train_learner_every": 1,
        "train_adversary_every": 2,
    }

    if dgp == "z_image":
        hp["learner_lr"] = 2e-4

    if dgp == "x_image":
        hp["learner_l2"] = 1e-8
        hp["adversary_l2"] = 1e-8

    return hp


# --------------------------------------------------
# Train/evaluate AGMM
# --------------------------------------------------
def run_agmm_on_data(data, dgp, device="cpu", DEBUG=False):
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
# Train/evaluate DeepGMM
# --------------------------------------------------
def run_deepgmm_on_data(data, dgp, device="cpu", DEBUG=False):
    """
    This script supports only dgp='z_image' because the current DeepGMM
    implementation uses scalar treatment T and image-valued instrument Z.
    """
    if dgp != "z_image":
        raise ValueError(
            "DeepGMM comparison currently supports only dgp='z_image'."
        )

    Z_train, T_train, Y_train, G_train = data[0]
    Z_val, T_val, Y_val, G_val = data[1]
    Z_test, T_test, Y_test, G_test = data[2]
    Z_dev, T_dev, Y_dev, G_dev = data[3]

    # Structural learner g uses scalar treatment.
    T_train_d = ensure_2d_column(as_tensor(T_train, device=device, dtype=torch.float64))
    Y_train_d = ensure_2d_column(as_tensor(Y_train, device=device, dtype=torch.float64))

    T_dev_d = ensure_2d_column(as_tensor(T_dev, device=device, dtype=torch.float64))
    Y_dev_d = ensure_2d_column(as_tensor(Y_dev, device=device, dtype=torch.float64))
    G_dev_d = ensure_2d_column(as_tensor(G_dev, device=device, dtype=torch.float64))

    T_test_d = ensure_2d_column(as_tensor(T_test, device=device, dtype=torch.float64))
    G_test_d = ensure_2d_column(as_tensor(G_test, device=device, dtype=torch.float64))

    # Adversary/test function uses image-valued instrument Z.
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
    Generate one dataset and run AGMM and DeepGMM on the same dataset.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"

    if dgp != "z_image":
        raise ValueError(
            "This final AGMM vs DeepGMM comparison supports only dgp='z_image'."
        )

    tau_code = sum(ord(ch) for ch in str(tau_fn))

    data_seed = int(
        base_seed
        + 100000 * run
        + 1000 * int(round(iv_strength * 10))
        + 17 * tau_code
        + int(num_data)
    )

    X_IMAGE, Z_IMAGE = dgp_to_bools(dgp)

    print("\nGenerating one shared dataset...")
    print("data_seed:", data_seed)

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

    print("\nTraining AGMM...")
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
        row["error"] = np.nan
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

    print("\nTraining DeepGMM...")
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
        row["error"] = np.nan
        row["elapsed_sec"] = time.time() - start_time

        print("DeepGMM result:", cleaned_results)

    except Exception as exc:
        row["error"] = str(exc)
        row["elapsed_sec"] = time.time() - start_time

        print("DeepGMM ERROR:", exc)

    rows.append(row)

    return rows


# --------------------------------------------------
# Summarize final results
# --------------------------------------------------
def summarize_results(raw_df):
    """
    Summarize successful Monte Carlo replications only.

    The summary focuses on performance metrics. Runtime and error columns are kept
    in the raw CSV but excluded from the performance summary.
    """
    id_cols = ["estimator", "dgp", "tau_fn", "iv_strength", "num_data"]

    df = raw_df.copy()

    if "error" in df.columns:
        df["success"] = df["error"].isna()
    else:
        df["success"] = True

    success_df = df[df["success"]].copy()

    exclude_cols = set(
        id_cols
        + [
            "run",
            "data_seed",
            "train_seed",
            "success",
            "elapsed_sec",
        ]
    )

    numeric_cols = success_df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [col for col in numeric_cols if col not in exclude_cols]

    if success_df.empty or not metric_cols:
        return pd.DataFrame()

    summary = (
        success_df
        .groupby(id_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

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

    success_counts = (
        success_df
        .groupby(id_cols, dropna=False)
        .size()
        .reset_index(name="MC_successful_runs")
    )

    summary = success_counts.merge(summary, on=id_cols, how="left")

    return summary


# --------------------------------------------------
# Main final thesis runner
# --------------------------------------------------
def main():
    device = "cpu"
    DEBUG = False

    print("\n========================================")
    print("Running FINAL thesis comparison: AGMM vs DeepGMM")
    print("========================================")
    print("device:", device)
    print("script_dir:", script_dir)
    print("project_root:", project_root)
    print("src_path:", src_path)
    print("deepgmm_path:", deepgmm_path)

    # --------------------------------------------------
    # Final thesis comparison settings
    # --------------------------------------------------
    tau_fn_list = ["abs", "sin"]
    iv_strength_list = [0.3,0.9]
    dgps = ["z_image"]
    num_data_list = [2000]
    monte_carlo = 3

    settings = list(
        itertools.product(
            tau_fn_list,
            iv_strength_list,
            dgps,
            num_data_list,
        )
    )

    print("\nFinal thesis configuration")
    print("tau_fn_list:", tau_fn_list)
    print("iv_strength_list:", iv_strength_list)
    print("dgps:", dgps)
    print("num_data_list:", num_data_list)
    print("monte_carlo:", monte_carlo)
    print("total settings:", len(settings))
    print("total paired runs:", len(settings) * monte_carlo)
    print("total model fits:", len(settings) * monte_carlo * 2)

    # --------------------------------------------------
    # Output paths
    # --------------------------------------------------
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    raw_path = os.path.join(
        results_dir,
        "thesis_agmm_deepgmm_comparison_raw_final.csv",
    )

    summary_path = os.path.join(
        results_dir,
        "thesis_agmm_deepgmm_comparison_summary_final.csv",
    )

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

            # Save checkpoint after every paired run.
            raw_df_checkpoint = pd.DataFrame(raw_rows)
            raw_df_checkpoint.to_csv(raw_path, index=False)

            summary_df_checkpoint = summarize_results(raw_df_checkpoint)
            summary_df_checkpoint.to_csv(summary_path, index=False)

            print("\nCheckpoint saved.")
            print("Raw results:", raw_path)
            print("Summary results:", summary_path)

    # --------------------------------------------------
    # Final save and print
    # --------------------------------------------------
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(raw_path, index=False)

    summary_df = summarize_results(raw_df)
    summary_df.to_csv(summary_path, index=False)

    print("\n---------- Finished FINAL thesis comparison ----------")
    print("Raw results saved to:", raw_path)
    print("Summary results saved to:", summary_path)

    print("\nRaw results:")
    print(raw_df)

    if not summary_df.empty:
        print("\nSummary results:")
        print(summary_df)
    else:
        print("\nNo successful runs to summarize.")


if __name__ == "__main__":
    main()