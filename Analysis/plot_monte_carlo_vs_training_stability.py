import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# --------------------------------------------------
# Project paths
# --------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
results_dir = project_root / "experiment" / "agmm" / "results"
output_dir = project_root / "Analysis"
output_dir.mkdir(exist_ok=True)


# --------------------------------------------------
# Load raw CSV files
# --------------------------------------------------
files = [
    results_dir / "iv_strength_linear_comparison_raw_final.csv",
    results_dir / "iv_strength_sin_comparison_raw_final.csv",
    results_dir / "iv_strength_abs_raw_final.csv",
]

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# --------------------------------------------------
# Keep required columns and clean types
# --------------------------------------------------
df["tau_fn"] = df["tau_fn"].astype(str)
df["iv_strength"] = df["iv_strength"].astype(float)
df["run"] = df["run"].astype(int)
df["R2_fin"] = df["R2_fin"].astype(float)


# --------------------------------------------------
# Compute cumulative Monte Carlo stability
# Stability metric:
# cumulative standard error of final R2
# lower values = more stable
# --------------------------------------------------
records = []

for (tau, iv), sub in df.groupby(["tau_fn", "iv_strength"]):
    sub = sub.sort_values("run")
    values = sub["R2_fin"].to_numpy()

    for k in range(2, len(values) + 1):
        current = values[:k]

        mean_r2 = np.mean(current)
        sd_r2 = np.std(current, ddof=1)
        se_r2 = sd_r2 / np.sqrt(k)

        records.append(
            {
                "tau_fn": tau,
                "iv_strength": iv,
                "mc_replications": k,
                "cumulative_mean_R2": mean_r2,
                "cumulative_sd_R2": sd_r2,
                "cumulative_se_R2": se_r2,
            }
        )

stability_df = pd.DataFrame(records)


# --------------------------------------------------
# Average stability across IV strengths for each tau_fn
# This produces one line per structural function
# --------------------------------------------------
plot_df = (
    stability_df.groupby(["tau_fn", "mc_replications"])
    .agg(avg_cumulative_se_R2=("cumulative_se_R2", "mean"))
    .reset_index()
)


# --------------------------------------------------
# Labels
# --------------------------------------------------
label_map = {
    "linear": "linear",
    "sin": "sin(x)",
    "abs": "|x|",
}
tau_order = ["linear", "sin", "abs"]


# --------------------------------------------------
# Plot 1: Monte Carlo replications vs training stability
# --------------------------------------------------
plt.figure(figsize=(7.5, 5.2))

for tau in tau_order:
    sub = plot_df[plot_df["tau_fn"] == tau].sort_values("mc_replications")

    plt.plot(
        sub["mc_replications"],
        sub["avg_cumulative_se_R2"],
        marker="o",
        linewidth=2,
        label=label_map.get(tau, tau),
    )

plt.xlabel("Number of Monte Carlo replications")
plt.ylabel(r"Average cumulative standard error of final $R^2$")
plt.title(r"Monte Carlo replications versus training stability")
plt.legend(title="Structural function")
plt.tight_layout()

plot_path = output_dir / "monte_carlo_vs_training_stability.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()


# --------------------------------------------------
# Optional Plot 2:
# Detailed version with separate line for each IV strength
# --------------------------------------------------
plt.figure(figsize=(8, 5.5))

for tau in tau_order:
    tau_sub = stability_df[stability_df["tau_fn"] == tau]

    for iv in sorted(tau_sub["iv_strength"].unique()):
        sub = tau_sub[tau_sub["iv_strength"] == iv].sort_values("mc_replications")

        plt.plot(
            sub["mc_replications"],
            sub["cumulative_se_R2"],
            marker="o",
            linewidth=1.8,
            label=f"{label_map.get(tau, tau)}, π={iv}",
        )

plt.xlabel("Number of Monte Carlo replications")
plt.ylabel(r"Cumulative standard error of final $R^2$")
plt.title(r"Detailed Monte Carlo stability by IV strength")
plt.legend(fontsize=9)
plt.tight_layout()

detailed_plot_path = output_dir / "monte_carlo_vs_training_stability_detailed.png"
plt.savefig(detailed_plot_path, dpi=300, bbox_inches="tight")
plt.show()


# --------------------------------------------------
# Save tables
# --------------------------------------------------
stability_table_path = output_dir / "monte_carlo_training_stability_table.csv"
summary_table_path = output_dir / "monte_carlo_training_stability_summary.csv"

stability_df.to_csv(stability_table_path, index=False)
plot_df.to_csv(summary_table_path, index=False)

print("Saved files:")
print(plot_path)
print(detailed_plot_path)
print(stability_table_path)
print(summary_table_path)