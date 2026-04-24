import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load results
df = pd.read_csv(r"C:\Users\tanji\Vscode project\ML_Econometrics_Thesis\experiment\agmm\results\iv_strength_tau_function_comparison_raw_final.csv")

# Keep only the tau functions and IV strengths you want
tau_functions = ["sin", "abs"]
iv_strengths = [0.3, 0.6, 0.9]

df = df[
    (df["tau_fn"].isin(tau_functions)) &
    (df["iv_strength"].isin(iv_strengths))
]

plt.figure(figsize=(8, 5))

groups = []
positions = []
labels = []

box_width = 0.30
x_positions = np.arange(len(iv_strengths))

for i, strength in enumerate(iv_strengths):
    sin_data = df[
        (df["iv_strength"] == strength) &
        (df["tau_fn"] == "sin")
    ]["MSE_earlystop"]

    abs_data = df[
        (df["iv_strength"] == strength) &
        (df["tau_fn"] == "abs")
    ]["MSE_earlystop"]

    groups.append(sin_data)
    groups.append(abs_data)

    positions.append(x_positions[i] - box_width / 2)
    positions.append(x_positions[i] + box_width / 2)

# Create grouped boxplot
plt.boxplot(
    groups,
    positions=positions,
    widths=box_width,
    patch_artist=True
)

# X-axis labels: one label per IV strength
plt.xticks(x_positions, [str(s) for s in iv_strengths])

plt.title("AGMM MSE Comparison: sin vs abs Across Instrument Strengths")
plt.xlabel("Instrument Strength ($\\pi$)")
plt.ylabel("MSE (Early Stop)")

# Add manual legend
plt.plot([], [], label="$h_0(x)=\\sin(x)$")
plt.plot([], [], label="$h_0(x)=|x|$")
plt.legend(title="Structural function")

plt.tight_layout()

# Save plot
os.makedirs("results", exist_ok=True)
plt.savefig("results/tau_function_iv_strength_boxplot.png", dpi=300)

plt.show()