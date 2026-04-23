import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/iv_strength_comparison_summary_final.csv")
df = df.sort_values("iv_strength")

plt.figure(figsize=(7, 5))

plt.errorbar(
    df["iv_strength"],
    df["avg_MSEearlystop"],
    yerr=df["std_MSEearlystop"],
    fmt='o-',
    linewidth=2,
    markersize=8,
    capsize=6
)

for _, row in df.iterrows():
    plt.text(
        row["iv_strength"],
        row["avg_MSEearlystop"] + 0.002,
        f'{row["avg_MSEearlystop"]:.3f}',
        ha='center'
    )

plt.title("AGMM Performance Across Instrument Strengths")
plt.xlabel("Instrument Strength ($\\pi$)")
plt.ylabel("Mean Squared Error (Early Stop)")
plt.xticks([0.3, 0.6, 0.9])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/iv_strength_visual_plot.png", dpi=300)
plt.show()