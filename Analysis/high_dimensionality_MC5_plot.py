import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load raw results
df = pd.read_csv("experiment\\agmm\\results\\high_dimensionality_MC5_raw_final.csv")

# Clean labels for thesis plot
df["DGP"] = df["dgp"].map({
    "z_image": "High-dimensional Z",
    "x_image": "High-dimensional X"
})

df["Tau function"] = df["tau_fn"].map({
    "abs": r"$|x|$",
    "sin": r"$\sin(x)$"
})

# Create grouped positions
groups = [
    ("High-dimensional Z", r"$|x|$"),
    ("High-dimensional Z", r"$\sin(x)$"),
    ("High-dimensional X", r"$|x|$"),
    ("High-dimensional X", r"$\sin(x)$"),
]

data = [
    df[(df["DGP"] == dgp) & (df["Tau function"] == tau)]["MSE_earlystop"].values
    for dgp, tau in groups
]

positions = np.arange(len(groups))

plt.figure(figsize=(9, 5))

# Boxplot
plt.boxplot(
    data,
    positions=positions,
    widths=0.55,
    showfliers=False,
    patch_artist=False
)

# Add raw Monte Carlo points
for i, values in enumerate(data):
    jitter = np.random.normal(0, 0.04, size=len(values))
    plt.scatter(
        np.full(len(values), positions[i]) + jitter,
        values,
        alpha=0.8,
        s=45
    )

plt.yscale("log")

plt.xticks(
    positions,
    [f"{dgp}\n{tau}" for dgp, tau in groups]
)

plt.ylabel(r"Early-stopping MSE, $\log$ scale")
plt.xlabel("Design and structural function")
plt.title("AGMM performance under high-dimensional treatment and instrument designs")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("high_dimensionality_mse_boxplot_log.png", dpi=300, bbox_inches="tight")
plt.show()