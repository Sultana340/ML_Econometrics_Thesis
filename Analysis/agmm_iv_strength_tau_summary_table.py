import pandas as pd
from pathlib import Path

# Define project root
project_root = Path(__file__).resolve().parents[1]

# Input summary CSV
csv_path = project_root / "experiment" / "agmm" / "results" / "iv_strength_tau_function_comparison_summary_final.csv"

# Load data
df = pd.read_csv(csv_path)

# Keep only selected columns
table_df = df[
    [
        "tau_fn",
        "iv_strength",
        "avg_MSEearlystop",
        "std_MSEearlystop",
        "avg_R2fin",
        "std_R2fin",
    ]
].copy()

# Sort table by tau function and IV strength
table_df = table_df.sort_values(["tau_fn", "iv_strength"])

# Rename columns for nicer display
table_df = table_df.rename(
    columns={
        "tau_fn": "Structural function",
        "iv_strength": "IV strength",
        "avg_MSEearlystop": "Avg. MSE early stop",
        "std_MSEearlystop": "Std. MSE early stop",
        "avg_R2fin": "Avg. final R2",
        "std_R2fin": "Std. final R2",
    }
)

# Round values
table_df = table_df.round(3)

# Print table in terminal
print(table_df)

# Save as CSV
output_csv = project_root / "Analysis" / "agmm_iv_strength_tau_summary_table.csv"
table_df.to_csv(output_csv, index=False)

# Save as LaTeX table
output_tex = project_root / "Analysis" / "agmm_iv_strength_tau_summary_table.tex"
table_df.to_latex(
    output_tex,
    index=False,
    caption="AGMM structural recovery under varying IV strength and structural complexity",
    label="tab:agmm_iv_strength_tau_summary",
    float_format="%.3f",
)

print(f"Table saved to: {output_csv}")
print(f"LaTeX table saved to: {output_tex}")