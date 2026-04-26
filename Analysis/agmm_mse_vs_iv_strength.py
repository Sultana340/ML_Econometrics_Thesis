import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define project root
project_root = Path(__file__).resolve().parents[1]

# Path to summary CSV
csv_path = project_root / "experiment" / "agmm" / "results" / "iv_strength_tau_function_comparison_summary_final.csv"

# Load data
df = pd.read_csv(csv_path)

# Check columns
print(df.columns)
print(df.head())

# Unique tau functions
tau_list = df["tau_fn"].unique()

plt.figure(figsize=(7, 5))

for tau in tau_list:
    sub = df[df["tau_fn"] == tau].sort_values("iv_strength")

    plt.errorbar(
        sub["iv_strength"],
        sub["avg_MSEearlystop"],      # mean MSE
        yerr=sub["std_MSEearlystop"], # std of MSE
        marker="o",
        capsize=4,
        linewidth=2,
        label=tau
    )

plt.xlabel(r"IV strength $\pi$")
plt.ylabel("Mean MSE")
plt.title("AGMM: Mean MSE versus IV strength")
plt.legend(title="Structural function")
plt.tight_layout()

# Save figure
output_path = project_root / "Analysis" / "agmm_mse_vs_iv_strength.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()