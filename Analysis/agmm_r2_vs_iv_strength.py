import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define project root
# If this script is inside Analysis/, then project_root is one folder above Analysis
project_root = Path(__file__).resolve().parents[1]

# Correct CSV path
csv_path = project_root / "experiment" / "agmm" / "results" / "iv_strength_tau_function_comparison_summary_final.csv"

# Load summary file
df = pd.read_csv(csv_path)

# Check column names
print(df.columns)
print(df.head())

# Unique tau functions
tau_list = df["tau_fn"].unique()

plt.figure(figsize=(7, 5))

for tau in tau_list:
    sub = df[df["tau_fn"] == tau].sort_values("iv_strength")

    plt.errorbar(
        sub["iv_strength"],
        sub["avg_R2fin"],
        yerr=sub["std_R2fin"],
        marker="o",
        capsize=4,
        linewidth=2,
        label=tau
    )

plt.axhline(0, linewidth=1, linestyle="--")
plt.xlabel(r"IV strength $\pi$")
plt.ylabel(r"Final $R^2$")
plt.title(r"AGMM: Final $R^2$ versus IV strength")
plt.legend(title="Structural function")
plt.tight_layout()

# Save figure
output_path = project_root / "Analysis" / "agmm_r2_vs_iv_strength.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()