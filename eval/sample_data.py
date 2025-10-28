import pandas as pd
from pathlib import Path

n = 13

# Resolve paths relative to this script's directory
base_dir = Path(__file__).parent
input_csv = base_dir / "browse_comp_decoded.csv"
output_csv = base_dir / "browse_comp_sample.csv"

# Load the full test set
df = pd.read_csv(input_csv)

# Randomly sample 12 rows
sample_df = df.sample(n)

# Save the sample to a new CSV
sample_df.to_csv(output_csv, index=False)
