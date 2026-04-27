import pandas as pd
import matplotlib.pyplot as plt

# Create dataframe from your table
df = pd.DataFrame({
    "tau_fn": ["x", "x", "x", "|x|", "|x|", "|x|", "sin(x)", "sin(x)", "sin(x)"],
    "iv_strength": [0.3, 0.6, 0.9, 0.3, 0.6, 0.9, 0.3, 0.6, 0.9],
    "avg_MSEearlystop": [0.006, 0.015, 0.015, 0.101, 0.029, 0.017, 0.079, 0.029, 0.022],
    "std_MSEearlystop": [0.003, 0.015, 0.015, 0.109, 0.018, 0.021, 0.029, 0.013, 0.012],
    "avg_R2fin": [0.950, 0.988, 0.983, 0.535, 0.925, 0.973, -0.389, 0.735, 0.847],
    "std_R2fin": [0.019, 0.015, 0.015, 0.162, 0.051, 0.024, 0.214, 0.112, 0.064]
})

tau_order = ["x", "|x|", "sin(x)"]

# -------------------------------
# Plot 1: Mean MSE early stop
# -------------------------------
plt.figure(figsize=(7, 5))

for tau in tau_order:
    sub = df[df["tau_fn"] == tau].sort_values("iv_strength")
    plt.errorbar(
        sub["iv_strength"],
        sub["avg_MSEearlystop"],
        yerr=sub["std_MSEearlystop"],
        marker="o",
        capsize=4,
        label=tau
    )

plt.xlabel("IV strength")
plt.ylabel("Mean MSE early stop")
plt.title("AGMM performance: MSE by IV strength and structural function")
plt.xticks([0.3, 0.6, 0.9])
plt.legend(title="Structural function")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("agmm_mse_iv_strength_plot.png", dpi=300)
plt.show()

# -------------------------------
# Plot 2: Mean final R^2
# -------------------------------
plt.figure(figsize=(7, 5))

for tau in tau_order:
    sub = df[df["tau_fn"] == tau].sort_values("iv_strength")
    plt.errorbar(
        sub["iv_strength"],
        sub["avg_R2fin"],
        yerr=sub["std_R2fin"],
        marker="o",
        capsize=4,
        label=tau
    )

plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("IV strength")
plt.ylabel("Mean final $R^2$")
plt.title("AGMM performance: final $R^2$ by IV strength and structural function")
plt.xticks([0.3, 0.6, 0.9])
plt.legend(title="Structural function")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("agmm_r2_iv_strength_plot.png", dpi=300)
plt.show()