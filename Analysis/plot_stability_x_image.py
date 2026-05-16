import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# --------------------------------------------------
# Project paths
# --------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
results_dir = project_root / "experiment" / "agmm" / "results"
output_dir = project_root / "Analysis" / "x_image_stability_check"
output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load high-dimensionality raw result file
# --------------------------------------------------
file_path = results_dir / "high_dimensionality_MC5_raw_final.csv"

df = pd.read_csv(file_path)


# --------------------------------------------------
# Keep only AGMM + x_image results
# --------------------------------------------------
df = df[
    (df["estimator"] == "AGMM") &
    (df["dgp"] == "x_image")
].copy()


# --------------------------------------------------
# Clean column types
# --------------------------------------------------
df["tau_fn"] = df["tau_fn"].astype(str)
df["iv_strength"] = df["iv_strength"].astype(float)
df["run"] = df["run"].astype(int)
df["R2_fin"] = df["R2_fin"].astype(float)
df["MSE_earlystop"] = df["MSE_earlystop"].astype(float)
df["MSE_fin"] = df["MSE_fin"].astype(float)


# --------------------------------------------------
# Compute cumulative Monte Carlo stability
# Stability metric:
# cumulative standard error of final R2 and early-stopping MSE
# lower values = more stable
# --------------------------------------------------
records = []

for (tau, iv), sub in df.groupby(["tau_fn", "iv_strength"]):
    sub = sub.sort_values("run")

    r2_values = sub["R2_fin"].to_numpy()
    mse_values = sub["MSE_earlystop"].to_numpy()

    for k in range(2, len(sub) + 1):
        current_r2 = r2_values[:k]
        current_mse = mse_values[:k]

        records.append(
            {
                "tau_fn": tau,
                "iv_strength": iv,
                "mc_replications": k,

                "cumulative_mean_R2_fin": np.mean(current_r2),
                "cumulative_sd_R2_fin": np.std(current_r2, ddof=1),
                "cumulative_se_R2_fin": np.std(current_r2, ddof=1) / np.sqrt(k),

                "cumulative_mean_MSE_earlystop": np.mean(current_mse),
                "cumulative_sd_MSE_earlystop": np.std(current_mse, ddof=1),
                "cumulative_se_MSE_earlystop": np.std(current_mse, ddof=1) / np.sqrt(k),
            }
        )

stability_df = pd.DataFrame(records)


# --------------------------------------------------
# Labels
# --------------------------------------------------
label_map = {
    "linear": "linear",
    "sin": "sin(x)",
    "abs": "|x|",
}

tau_order = ["abs", "sin"]


# --------------------------------------------------
# Plot 1: Cumulative standard error of final R2
# --------------------------------------------------
plt.figure(figsize=(7.5, 5.2))

for tau in tau_order:
    sub = stability_df[stability_df["tau_fn"] == tau].sort_values("mc_replications")

    if sub.empty:
        continue

    plt.plot(
        sub["mc_replications"],
        sub["cumulative_se_R2_fin"],
        marker="o",
        linewidth=2,
        label=label_map.get(tau, tau),
    )

plt.xlabel("Number of Monte Carlo replications")
plt.ylabel(r"Cumulative standard error of final $R^2$")
plt.title(r"AGMM stability check for $x\_image$: final $R^2$")
plt.yscale("log")
plt.legend(title="Structural function")
plt.tight_layout()

plot_r2_se_path = output_dir / "x_image_cumulative_se_R2_fin.png"
plt.savefig(plot_r2_se_path, dpi=300, bbox_inches="tight")
plt.show()


# --------------------------------------------------
# Plot 2: Cumulative standard error of early-stopping MSE
# --------------------------------------------------
plt.figure(figsize=(7.5, 5.2))

for tau in tau_order:
    sub = stability_df[stability_df["tau_fn"] == tau].sort_values("mc_replications")

    if sub.empty:
        continue

    plt.plot(
        sub["mc_replications"],
        sub["cumulative_se_MSE_earlystop"],
        marker="o",
        linewidth=2,
        label=label_map.get(tau, tau),
    )

plt.xlabel("Number of Monte Carlo replications")
plt.ylabel(r"Cumulative standard error of early-stopping MSE")
plt.title(r"AGMM stability check for $x\_image$: early-stopping MSE")
plt.yscale("log")
plt.legend(title="Structural function")
plt.tight_layout()

plot_mse_se_path = output_dir / "x_image_cumulative_se_MSE_earlystop.png"
plt.savefig(plot_mse_se_path, dpi=300, bbox_inches="tight")
plt.show()


# --------------------------------------------------
# Plot 3: Final R2 across Monte Carlo runs
# Useful for detecting catastrophic runs
# --------------------------------------------------
plt.figure(figsize=(7.5, 5.2))

for tau in tau_order:
    sub = df[df["tau_fn"] == tau].sort_values("run")

    if sub.empty:
        continue

    plt.plot(
        sub["run"],
        sub["R2_fin"],
        marker="o",
        linewidth=2,
        label=label_map.get(tau, tau),
    )

plt.xlabel("Monte Carlo run")
plt.ylabel(r"Final $R^2$")
plt.title(r"AGMM $x\_image$ performance across Monte Carlo runs")
plt.yscale("symlog")
plt.legend(title="Structural function")
plt.tight_layout()

plot_r2_runs_path = output_dir / "x_image_R2_fin_by_run.png"
plt.savefig(plot_r2_runs_path, dpi=300, bbox_inches="tight")
plt.show()


# --------------------------------------------------
# Summary table by tau function
# --------------------------------------------------
summary_df = (
    df.groupby(["dgp", "tau_fn", "iv_strength", "estimator"])
    .agg(
        MC_runs=("run", "count"),
        mean_R2_fin=("R2_fin", "mean"),
        std_R2_fin=("R2_fin", "std"),
        median_R2_fin=("R2_fin", "median"),
        mean_MSE_earlystop=("MSE_earlystop", "mean"),
        std_MSE_earlystop=("MSE_earlystop", "std"),
        median_MSE_earlystop=("MSE_earlystop", "median"),
        mean_MSE_fin=("MSE_fin", "mean"),
        std_MSE_fin=("MSE_fin", "std"),
        median_MSE_fin=("MSE_fin", "median"),
    )
    .reset_index()
)


# --------------------------------------------------
# Save tables
# --------------------------------------------------
stability_table_path = output_dir / "x_image_monte_carlo_stability_table.csv"
summary_table_path = output_dir / "x_image_stability_summary.csv"

stability_df.to_csv(stability_table_path, index=False)
summary_df.to_csv(summary_table_path, index=False)


print("Saved files:")
print(plot_r2_se_path)
print(plot_mse_se_path)
print(plot_r2_runs_path)
print(stability_table_path)
print(summary_table_path)

print("\nSummary:")
print(summary_df)