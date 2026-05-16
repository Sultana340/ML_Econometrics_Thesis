import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Change this path if your CSV is in another folder
csv_path = project_root / "experiment" / "agmm" / "results" / "robustness_centroid_ncenters_raw_abs_iv06.csv"

# If the CSV is inside the Analysis folder, use this instead:
# csv_path = script_dir / "robustness_centroid_ncenters_raw_abs_iv06.csv"

# --------------------------------------------------
# Load CSV
# --------------------------------------------------
df = pd.read_csv(csv_path)

# --------------------------------------------------
# Keep valid rows only
# --------------------------------------------------
df = df.dropna(subset=["n_centers", "MSE_earlystop"]).copy()
df = df.sort_values(["n_centers", "run"])

# --------------------------------------------------
# Compute mean and standard deviation by n_centers
# --------------------------------------------------
summary = (
    df.groupby("n_centers")
    .agg(
        mean_mse=("MSE_earlystop", "mean"),
        std_mse=("MSE_earlystop", "std"),
        mc_runs=("run", "count")
    )
    .reset_index()
)

print(summary)

# --------------------------------------------------
# Plot: Mean MSE with standard-deviation error bars
# --------------------------------------------------
plt.figure(figsize=(7, 5))

plt.bar(
    summary["n_centers"].astype(str),
    summary["mean_mse"],
    yerr=summary["std_mse"],
    capsize=6,
    alpha=0.8
)

# Add value labels above bars
for i, row in summary.iterrows():
    plt.text(
        i,
        row["mean_mse"] + row["std_mse"] + 0.08,
        f'{row["mean_mse"]:.3f}',
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xlabel(r"Number of centroids, $n_{\mathrm{centers}}$")
plt.ylabel(r"Mean early-stopping MSE")
plt.title(r"Robustness Check: Mean MSE across $n_{\mathrm{centers}}$")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

# --------------------------------------------------
# Save figure
# --------------------------------------------------
output_path = script_dir / "robustness_centroid_ncenters_mse_barplot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {output_path}")