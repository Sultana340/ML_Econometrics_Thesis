import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/iv_strength_comparison_raw_final.csv")

plt.figure(figsize=(7, 5))

groups = []
labels = []

for strength in sorted(df["iv_strength"].unique()):
    groups.append(df[df["iv_strength"] == strength]["MSE_earlystop"])
    labels.append(str(strength))

plt.boxplot(groups, tick_labels=labels)

plt.title("Distribution of AGMM MSE Across Monte Carlo Runs")
plt.xlabel("Instrument Strength ($\\pi$)")
plt.ylabel("MSE (Early Stop)")
plt.tight_layout()
plt.savefig("results/iv_strength_boxplot.png", dpi=300)
plt.show()