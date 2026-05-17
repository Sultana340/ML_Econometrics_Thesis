import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

results_dir = project_root / "experiment" / "agmm" / "results"
output_dir = project_root / "Analysis"
output_dir.mkdir(exist_ok=True)

file_linear = results_dir / "high_dimensionality_MC5_raw_final.csv"
file_abs_sin = results_dir / "high_dimensionality_comparison_raw_final.csv"

print("Looking for:", file_linear)
print("Exists:", file_linear.exists())

print("Looking for:", file_abs_sin)
print("Exists:", file_abs_sin.exists())


# --------------------------------------------------
# Read and combine data
# --------------------------------------------------
df_linear = pd.read_csv(file_linear)
df_abs_sin = pd.read_csv(file_abs_sin)

df = pd.concat([df_linear, df_abs_sin], ignore_index=True)


# --------------------------------------------------
# Keep only x_image and the three structural functions
# --------------------------------------------------
x_df = df[
    (df["dgp"] == "x_image") &
    (df["tau_fn"].isin(["linear", "abs", "sin"]))
].copy()

# Optional: keep only AGMM if estimator column exists
if "estimator" in x_df.columns:
    x_df = x_df[x_df["estimator"] == "AGMM"].copy()

x_df = x_df.sort_values(["tau_fn", "run"])

print("\nAvailable x_image observations:")
print(x_df.groupby("tau_fn")["run"].count())


# --------------------------------------------------
# Compute cumulative training stability
# Stability = cumulative standard error of MSE_earlystop
# --------------------------------------------------
records = []

for tau_fn, group in x_df.groupby("tau_fn"):
    group = group.sort_values("run")
    mse_values = group["MSE_earlystop"].to_numpy(dtype=float)

    for k in range(1, len(mse_values) + 1):
        current_values = mse_values[:k]

        if k == 1:
            cumulative_sd = np.nan
            cumulative_se = np.nan
        else:
            cumulative_sd = np.std(current_values, ddof=1)
            cumulative_se = cumulative_sd / np.sqrt(k)

        records.append({
            "tau_fn": tau_fn,
            "mc_replications": k,
            "cumulative_mean_mse": np.mean(current_values),
            "cumulative_sd_mse": cumulative_sd,
            "cumulative_se_mse": cumulative_se,
        })

stability_df = pd.DataFrame(records)

summary_path = output_dir / "x_image_mc_training_stability_summary.csv"
stability_df.to_csv(summary_path, index=False)

print("\nSaved summary CSV to:")
print(summary_path)


# --------------------------------------------------
# Plot Monte Carlo replications vs training stability
# --------------------------------------------------
plt.figure(figsize=(7.2, 4.6))

for tau_fn in ["linear", "abs", "sin"]:
    plot_df = stability_df[
        (stability_df["tau_fn"] == tau_fn) &
        (stability_df["mc_replications"] >= 2)
    ]

    if plot_df.empty:
        continue

    plt.plot(
        plot_df["mc_replications"],
        plot_df["cumulative_se_mse"],
        marker="o",
        linewidth=2,
        label=tau_fn
    )

plt.xlabel("Monte Carlo replications")
plt.ylabel(r"Cumulative standard error of MSE$_{\mathrm{early}}$")
plt.title("Training stability under the x_image design")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.35)
plt.legend(title="Structural function")
plt.tight_layout()

figure_path = output_dir / "x_image_mc_replications_training_stability.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure to:")
print(figure_path)