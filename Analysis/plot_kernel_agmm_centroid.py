import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Automatically find the CSV files inside the project
abs_matches = list(project_root.rglob("Final_variant_MC5_raw_abs.csv"))
sin_matches = list(project_root.rglob("Final_variant_MC5_raw_sin.csv"))

if len(abs_matches) == 0:
    raise FileNotFoundError("Could not find Final_variant_MC5_raw_abs.csv")

if len(sin_matches) == 0:
    raise FileNotFoundError("Could not find Final_variant_MC5_raw_sin.csv")

abs_path = abs_matches[0]
sin_path = sin_matches[0]

output_path = script_dir / "mse_early_stability_by_structural_function.png"

# --------------------------------------------------
# Load data
# --------------------------------------------------
abs_df = pd.read_csv(abs_path)
sin_df = pd.read_csv(sin_path)

data = pd.concat([abs_df, sin_df], ignore_index=True)

# --------------------------------------------------
# Settings
# --------------------------------------------------
estimators = ["AGMM", "KernelLayerMMDGMM", "CentroidMMDGMM"]

tau_labels = {
    "abs": r"$|x|$",
    "sin": r"$\sin(x)$"
}

tau_order = ["abs", "sin"]

# --------------------------------------------------
# Create figure
# One subplot for each structural function
# Left panel  = |x|
# Right panel = sin(x)
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for ax, tau in zip(axes, tau_order):

    # Select data for one structural function
    tau_df = data[data["tau_fn"] == tau]

    for est in estimators:

        # Select data for one estimator
        sub = tau_df[tau_df["estimator"] == est].sort_values("run")

        # --------------------------------------------------
        # Solid line:
        # MSE_earlystop for each Monte Carlo replication
        # --------------------------------------------------
        line, = ax.plot(
            sub["run"],
            sub["MSE_earlystop"],
            marker="o",
            linewidth=2,
            label=est
        )

        # Get the same color as the estimator line
        color = line.get_color()

        # --------------------------------------------------
        # Dashed horizontal line:
        # Mean MSE_earlystop across all Monte Carlo runs
        # for the same estimator and structural function
        # --------------------------------------------------
        mean_mse = sub["MSE_earlystop"].mean()

        ax.axhline(
            mean_mse,
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            color=color
        )

    # --------------------------------------------------
    # Axis formatting
    # --------------------------------------------------
    ax.set_title(fr"Structural function: {tau_labels[tau]}")
    ax.set_xlabel("Monte Carlo run")
    ax.set_xticks(sorted(tau_df["run"].unique()))
    ax.set_ylabel(r"MSE$_{\mathrm{early}}$")
    ax.grid(alpha=0.3)

    # --------------------------------------------------
    # Small note inside each subplot
    # --------------------------------------------------
    ax.text(
        0.02,
        0.96,
        "Solid line: run-wise MSE\nDashed line: MC mean",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

# --------------------------------------------------
# Legend 1:
# Estimator legend
# Color indicates estimator
# --------------------------------------------------
estimator_handles, estimator_labels = axes[0].get_legend_handles_labels()

legend1 = fig.legend(
    estimator_handles,
    estimator_labels,
    title="Estimator",
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.08),
    frameon=False
)

# --------------------------------------------------
# Legend 2:
# Line-style legend
# Solid line  = Monte Carlo run-wise values
# Dashed line = Monte Carlo mean
# --------------------------------------------------
style_handles = [
    Line2D([0], [0], color="black", linewidth=2, linestyle="-",
           label="Run-wise MSE"),
    Line2D([0], [0], color="black", linewidth=1.5, linestyle="--",
           label="Monte Carlo mean")
]

fig.legend(
    handles=style_handles,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.00),
    frameon=False
)

# --------------------------------------------------
# Figure title
# --------------------------------------------------
fig.suptitle(
    r"Monte Carlo Stability of MSE$_{\mathrm{early}}$ across Structural Functions",
    fontsize=14,
    y=1.18
)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {output_path}")