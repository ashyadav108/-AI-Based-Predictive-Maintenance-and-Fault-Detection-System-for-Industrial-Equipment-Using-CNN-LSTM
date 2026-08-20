"""
=========================================================
NASA C-MAPSS RUL Prediction
File        : preprocess.py
Purpose     : Load, validate and explore the dataset
Author      : Your Name
=========================================================
"""

# =====================================================
# Import Libraries
# =====================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =====================================================
# Create Output Folder
# =====================================================

os.makedirs("../results", exist_ok=True)

# =====================================================
# Dataset Paths
# =====================================================

TRAIN_PATH = "../dataset/train_FD001.txt"
TEST_PATH = "../dataset/test_FD001.txt"
RUL_PATH = "../dataset/RUL_FD001.txt"

# =====================================================
# Column Names
# =====================================================

columns = [
    "engine_id",
    "cycle",
    "op_setting1",
    "op_setting2",
    "op_setting3",
]

for i in range(1, 22):
    columns.append(f"sensor_{i}")

# =====================================================
# Function : Load Dataset
# =====================================================

def load_dataset(path):
    """
    Load NASA C-MAPSS dataset.
    """

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    # Remove extra empty columns if present
    df.drop(columns=[26, 27], inplace=True, errors="ignore")

    # Assign column names
    df.columns = columns

    return df

# =====================================================
# Load All Files
# =====================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

train_df = load_dataset(TRAIN_PATH)
test_df = load_dataset(TEST_PATH)

rul_df = pd.read_csv(
    RUL_PATH,
    header=None
)

rul_df.columns = ["RUL"]

print("Datasets Loaded Successfully.\n")

# =====================================================
# Dataset Shapes
# =====================================================

print("=" * 60)
print("Dataset Shapes")
print("=" * 60)

print(f"Training Dataset : {train_df.shape}")
print(f"Testing Dataset  : {test_df.shape}")
print(f"RUL Dataset      : {rul_df.shape}")

# =====================================================
# Display First Five Rows
# =====================================================

print("\n")
print("=" * 60)
print("Training Dataset")
print("=" * 60)

print(train_df.head())

print("\n")
print("=" * 60)
print("Testing Dataset")
print("=" * 60)

print(test_df.head())

print("\n")
print("=" * 60)
print("RUL Dataset")
print("=" * 60)

print(rul_df.head())

# =====================================================
# Dataset Information
# =====================================================

print("\n")
print("=" * 60)
print("Training Dataset Information")
print("=" * 60)

print(train_df.info())

# =====================================================
# Missing Values
# =====================================================

print("\n")
print("=" * 60)
print("Missing Values")
print("=" * 60)

print(train_df.isnull().sum())

# =====================================================
# Duplicate Rows
# =====================================================

print("\n")
print("=" * 60)
print("Duplicate Rows")
print("=" * 60)

duplicates = train_df.duplicated().sum()

print(f"Duplicate Rows : {duplicates}")

# =====================================================
# Summary Statistics
# =====================================================

print("\n")
print("=" * 60)
print("Summary Statistics")
print("=" * 60)

print(train_df.describe())

# =====================================================
# Number of Engines
# =====================================================

print("\n")
print("=" * 60)
print("Number of Engines")
print("=" * 60)

num_engines = train_df["engine_id"].nunique()

print(f"Total Engines : {num_engines}")

# =====================================================
# Engine Life
# =====================================================

engine_life = (
    train_df
    .groupby("engine_id")["cycle"]
    .max()
)

print("\n")
print("=" * 60)
print("Engine Life Statistics")
print("=" * 60)

print(engine_life.describe())

print(f"\nMinimum Engine Life : {engine_life.min()} cycles")
print(f"Maximum Engine Life : {engine_life.max()} cycles")
print(f"Average Engine Life : {engine_life.mean():.2f} cycles")

# =====================================================
# Engine Life Distribution
# =====================================================

plt.figure(figsize=(10,6))

plt.hist(
    engine_life,
    bins=20,
    edgecolor="black"
)

plt.title("Distribution of Engine Life")
plt.xlabel("Maximum Cycle")
plt.ylabel("Number of Engines")

plt.tight_layout()

plt.savefig("../results/engine_life_distribution.png")

plt.close()

print("\nEngine life distribution saved.")

# =====================================================
# Sensor Variability
# =====================================================

sensor_columns = [f"sensor_{i}" for i in range(1,22)]

sensor_std = train_df[sensor_columns].std()

print("\n")
print("=" * 60)
print("Sensor Standard Deviation")
print("=" * 60)

print(sensor_std.sort_values())

# =====================================================
# End of Part 1
# =====================================================

print("\n")
print("=" * 60)
print("Preprocessing Part 1 Completed Successfully")
print("=" * 60)

# =====================================================
# Generate RUL Labels
# =====================================================

print("\n" + "=" * 60)
print("Generating Remaining Useful Life (RUL)")
print("=" * 60)

# Maximum cycle reached by every engine
max_cycle = (
    train_df
    .groupby("engine_id")["cycle"]
    .max()
    .reset_index()
)

max_cycle.columns = ["engine_id", "max_cycle"]

# Merge with training dataframe
train_df = train_df.merge(
    max_cycle,
    on="engine_id",
    how="left"
)

# Calculate RUL
train_df["RUL"] = (
    train_df["max_cycle"] -
    train_df["cycle"]
)

print("RUL Generated Successfully.")

print("\nSample RUL")

print(
    train_df[
        ["engine_id","cycle","max_cycle","RUL"]
    ].head(15)
)

# =====================================================
# RUL Statistics
# =====================================================

print("\n" + "=" * 60)
print("RUL Statistics")
print("=" * 60)

print(train_df["RUL"].describe())

# =====================================================
# Cap RUL
# =====================================================

RUL_CAP = 125

train_df["RUL"] = train_df["RUL"].clip(
    upper=RUL_CAP
)

print(f"\nMaximum RUL after capping : {train_df['RUL'].max()}")

# =====================================================
# Plot RUL Distribution
# =====================================================

plt.figure(figsize=(10,6))

plt.hist(
    train_df["RUL"],
    bins=30,
    edgecolor="black"
)

plt.title("RUL Distribution")
plt.xlabel("Remaining Useful Life")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("../results/rul_distribution.png")

plt.close()

print("RUL distribution saved.")

# =====================================================
# Correlation Heatmap
# =====================================================

print("\nCreating Correlation Heatmap...")

corr = train_df.corr(numeric_only=True)

plt.figure(figsize=(18,14))

plt.imshow(
    corr,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns,
    fontsize=7
)

plt.tight_layout()

plt.savefig("../results/correlation_heatmap.png")

plt.close()

print("Correlation heatmap saved.")

# =====================================================
# Save Processed Dataset
# =====================================================

os.makedirs("../dataset/processed", exist_ok=True)

train_df.to_csv(
    "../dataset/processed/train_processed.csv",
    index=False
)

test_df.to_csv(
    "../dataset/processed/test_processed.csv",
    index=False
)

rul_df.to_csv(
    "../dataset/processed/rul_processed.csv",
    index=False
)

print("\nProcessed datasets saved successfully.")

# =====================================================
# Final Dataset Information
# =====================================================

print("\n" + "=" * 60)
print("Final Dataset Shapes")
print("=" * 60)

print("Train :", train_df.shape)
print("Test  :", test_df.shape)
print("RUL   :", rul_df.shape)

print("\nColumns in Train Dataset")

print(train_df.columns)

# =====================================================
# Preprocessing Completed
# =====================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

