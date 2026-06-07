import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_FILE = "data/raw/lmd_dataset.csv"
OUTPUT_FILE = "data/processed/processed_data.csv"

print("Loading dataset...")
df = pd.read_csv(RAW_FILE)

print(f"Original Shape: {df.shape}")

drop_cols = [
    "SystemTime",
    "Unnamed: 0.2"
]

existing_cols = [col for col in drop_cols if col in df.columns]

df = df.drop(columns=existing_cols)

print(f"Shape after cleanup: {df.shape}")

normal = df[df["Label"] == 0]
eoht = df[df["Label"] == 1]
eors = df[df["Label"] == 2]

print("\nClass Distribution:")
print(df["Label"].value_counts())

sample_size = 15000

normal_sample = normal.sample(
    n=min(sample_size, len(normal)),
    random_state=42
)

eoht_sample = eoht.sample(
    n=min(sample_size, len(eoht)),
    random_state=42
)

eors_sample = eors.sample(
    n=min(sample_size, len(eors)),
    random_state=42
)

balanced_df = pd.concat(
    [normal_sample, eoht_sample, eors_sample]
)

balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nBalanced Distribution:")
print(balanced_df["Label"].value_counts())

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

balanced_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved: {OUTPUT_FILE}")
print("Final Shape:", balanced_df.shape)