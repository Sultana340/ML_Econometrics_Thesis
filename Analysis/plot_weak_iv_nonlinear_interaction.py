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
# Load raw result files
# --------------------------------------------------
files = [
    results_dir / "iv_strength_linear_comparison_raw_final.csv",
    results_dir / "iv_strength_sin_comparison_raw_final.csv",
    results_dir / "iv_strength_abs_raw_final.csv",
]

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Make sure key columns have correct types
df["iv_strength"] = df["iv_strength"].astype(float)
df["tau_fn"] = df["tau_fn"].astype(str)
df["R2_fin"] = df["R2_fin"].astype(float)

# Optional: keep only relevant tau functions
df = df[df["tau_fn"].isin(["linear", "sin", "abs"])].copy()


# --------------------------------------------------
# Thesis-friendly labels
# --------------------------------------------------
label_map = {
    "linear": "Linear",
    "sin": r"Nonlinear: $\sin(x)$",
    "abs": r"Nonsmooth nonlinear: $|x|$",
}

tau_order = ["linear", "sin", "abs"]


# --------------------------------------------------
# Summary statistics
# --------------------------------------------------
summary = (
    df.groupby(["tau_fn", "iv_strength"])
    .agg(
        mean_R2=("R2_fin", "mean"),
        std_R2=("R2_fin", "std"),
        n_runs=("R2_fin", "count"),
    )
    .reset_index()
)


# ==================================================
# FIGURE 1:
# Weak-IV and nonlinear-function interaction
# Mean R2 with raw Monte Carlo points
# ==================================================
plt.figure(figsize=(8, 5.5))

offset_map = {
    "linear": -0.015,
    "sin": 0.0,
    "abs": 0.015,
}

for tau in tau_order:
    raw_sub = df[df["tau_fn"] == tau].copy()
    sum_sub = summary[summary["tau_fn"] == tau].sort_values("iv_strength")

    # Plot raw Monte Carlo points with small jitter
    for iv in sorted(raw_sub["iv_strength"].unique()):
        y = raw_sub.loc[raw_sub["iv_strength"] == iv, "R2_fin"].values

        if len(y) > 1:
            jitter = np.linspace(-0.006, 0.006, len(y))
        else:
            jitter = np.array([0.0])

        x = iv + offset_map[tau] + jitter

        plt.scatter(
            x,
            y,
            s=35,
            alpha=0.8,
        )

    # Mean ± std line
    plt.errorbar(
        sum_sub["iv_strength"] + offset_map[tau],
        sum_sub["mean_R2"],
        yerr=sum_sub["std_R2"],
        marker="o",
        linewidth=2,
        capsize=4,
        label=label_map[tau],
    )

# Mark weak-IV region
plt.axvspan(0.25, 0.35, alpha=0.12)
plt.axhline(0, linewidth=1, linestyle="--")

plt.xlabel(r"IV strength $\pi$")
plt.ylabel(r"Final $R^2$")
plt.title(r"Weak-instrument and nonlinear-function interaction")
plt.legend(title="Structural function")
plt.tight_layout()

fig1_path = output_dir / "weak_iv_nonlinear_interaction_r2.png"
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# FIGURE 2:
# Nonlinear penalty relative to linear benchmark
#
# penalty = mean R2(linear) - mean R2(nonlinear)
#
# Larger positive values mean the nonlinear function is harder
# to recover relative to the linear benchmark.
# ==================================================
wide = summary.pivot(index="iv_strength", columns="tau_fn", values="mean_R2")

penalty_df = pd.DataFrame(
    {
        "iv_strength": wide.index,
        "sin_penalty": wide["linear"] - wide["sin"],
        "abs_penalty": wide["linear"] - wide["abs"],
    }
).reset_index(drop=True)

plt.figure(figsize=(7, 5))

plt.plot(
    penalty_df["iv_strength"],
    penalty_df["sin_penalty"],
    marker="o",
    linewidth=2,
    label=r"Penalty: linear $-$ $\sin(x)$",
)

plt.plot(
    penalty_df["iv_strength"],
    penalty_df["abs_penalty"],
    marker="o",
    linewidth=2,
    label=r"Penalty: linear $-$ $|x|$",
)

plt.axhline(0, linewidth=1, linestyle="--")
plt.axvspan(0.25, 0.35, alpha=0.12)

plt.xlabel(r"IV strength $\pi$")
plt.ylabel(r"Nonlinear recovery penalty in final $R^2$")
plt.title("Nonlinear penalty under weak and strong instruments")
plt.legend()
plt.tight_layout()

fig2_path = output_dir / "nonlinear_penalty_by_iv_strength.png"
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.show()


# --------------------------------------------------
# Save interaction summary table
# --------------------------------------------------
interaction_table = summary.copy()
interaction_table["structural_function"] = interaction_table["tau_fn"].map(label_map)

interaction_table = interaction_table[
    [
        "structural_function",
        "iv_strength",
        "mean_R2",
        "std_R2",
        "n_runs",
    ]
]

interaction_table = interaction_table.round(4)

table_path = output_dir / "weak_iv_nonlinear_interaction_summary.csv"
interaction_table.to_csv(table_path, index=False)

penalty_path = output_dir / "nonlinear_penalty_by_iv_strength.csv"
penalty_df.round(4).to_csv(penalty_path, index=False)

print("Saved figures and tables:")
print(fig1_path)
print(fig2_path)
print(table_path)
print(penalty_path)