"""
=========================================================
NASA C-MAPSS Remaining Useful Life Prediction
File        : dataset.py
Description : Load processed dataset and perform
              feature scaling.
=========================================================
"""

# =========================================================
# Import Libraries
# =========================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# =========================================================
# Dataset Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "..", "dataset", "processed", "train_processed.csv")
TEST_PATH = os.path.join(BASE_DIR, "..", "dataset", "processed", "test_processed.csv")

# =========================================================
# Load Dataset
# =========================================================

print("=" * 60)
print("Loading Processed Dataset")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nTraining Dataset Shape :", train_df.shape)
print("Testing Dataset Shape  :", test_df.shape)

# =========================================================
# Display First Five Rows
# =========================================================

print("\nFirst Five Rows of Training Dataset\n")
print(train_df.head())

# =========================================================
# Select Input Features
# =========================================================

feature_columns = [

    "op_setting1",
    "op_setting2",
    "op_setting3",

    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"

]

target_column = "RUL"

# =========================================================
# Separate Features and Target
# =========================================================

X_train = train_df[feature_columns]

y_train = train_df[target_column]

X_test = test_df[feature_columns]

print("\nFeature Matrix Shape :", X_train.shape)
print("Target Vector Shape  :", y_train.shape)

# =========================================================
# Feature Scaling
# =========================================================

print("\nApplying MinMax Scaling...")

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Scaling Completed Successfully.")

# =========================================================
# Convert Back to DataFrame
# =========================================================

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=feature_columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=feature_columns
)

# =========================================================
# Add Required Columns
# =========================================================

X_train_scaled["engine_id"] = train_df["engine_id"].values
X_train_scaled["cycle"] = train_df["cycle"].values
X_train_scaled["RUL"] = train_df["RUL"].values

X_test_scaled["engine_id"] = test_df["engine_id"].values
X_test_scaled["cycle"] = test_df["cycle"].values

# =========================================================
# Verify Scaled Dataset
# =========================================================

print("\nScaled Training Dataset\n")
print(X_train_scaled.head())

print("\nScaled Testing Dataset\n")
print(X_test_scaled.head())

print("\nScaled Training Shape :", X_train_scaled.shape)
print("Scaled Testing Shape  :", X_test_scaled.shape)

# =========================================================
# Part 1 Completed
# =========================================================

print("\n" + "=" * 60)
print("DATASET PART 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

# =========================================================
# Sliding Window Parameters
# =========================================================

SEQUENCE_LENGTH = 30

# =========================================================
# Create Sliding Window Function
# =========================================================

def create_train_sequences(dataframe, feature_columns, sequence_length):

    X = []
    y = []

    # Process one engine at a time
    for engine_id in dataframe["engine_id"].unique():

        engine_data = dataframe[
            dataframe["engine_id"] == engine_id
        ].reset_index(drop=True)

        feature_data = engine_data[feature_columns].values
        rul_data = engine_data["RUL"].values

        # Skip engines with fewer cycles than sequence length
        if len(engine_data) < sequence_length:
            continue

        # Generate sequences
        for i in range(len(engine_data) - sequence_length + 1):

            X.append(
                feature_data[i:i + sequence_length]
            )

            y.append(
                rul_data[i + sequence_length - 1]
            )

    return np.array(X), np.array(y)


# =========================================================
# Generate Training Sequences
# =========================================================

print("\n" + "=" * 60)
print("Generating Training Sequences")
print("=" * 60)

X_train_seq, y_train_seq = create_train_sequences(
    X_train_scaled,
    feature_columns,
    SEQUENCE_LENGTH
)

print("Training Sequences Generated Successfully")

print("Input Shape :", X_train_seq.shape)
print("Target Shape:", y_train_seq.shape)

# =========================================================
# Create Test Sequences
# =========================================================

def create_test_sequences(dataframe, feature_columns, sequence_length):

    X_test = []

    for engine_id in dataframe["engine_id"].unique():

        engine_data = dataframe[
            dataframe["engine_id"] == engine_id
        ].reset_index(drop=True)

        feature_data = engine_data[feature_columns].values

        # If engine has fewer than 30 cycles, pad with zeros
        if len(feature_data) < sequence_length:

            padding = np.zeros(
                (
                    sequence_length - len(feature_data),
                    len(feature_columns)
                )
            )

            feature_data = np.vstack(
                (padding, feature_data)
            )

        else:
            # Take the last 30 cycles
            feature_data = feature_data[-sequence_length:]

        X_test.append(feature_data)

    return np.array(X_test)


# =========================================================
# Generate Test Sequences
# =========================================================

print("\n" + "=" * 60)
print("Generating Test Sequences")
print("=" * 60)

X_test_seq = create_test_sequences(
    X_test_scaled,
    feature_columns,
    SEQUENCE_LENGTH
)

print("Test Sequences Generated Successfully")

print("Test Shape :", X_test_seq.shape)

# =========================================================
# Verify Data
# =========================================================

print("\nSample Input Shape :", X_train_seq[0].shape)
print("Sample Target      :", y_train_seq[0])

print("\n" + "=" * 60)
print("DATASET PART 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

# =========================================================
# Import Required Libraries
# =========================================================

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

# =========================================================
# Train Validation Split
# =========================================================

print("\n" + "=" * 60)
print("Splitting Training and Validation Data")
print("=" * 60)

(
    X_train,
    X_val,
    y_train,
    y_val
) = train_test_split(
    X_train_seq,
    y_train_seq,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Training Samples  :", len(X_train))
print("Validation Samples:", len(X_val))

# =========================================================
# Convert to Tensor
# =========================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test_seq,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32
)

print("\nTensor Conversion Completed")

# =========================================================
# Custom Dataset
# =========================================================

class RULDataset(Dataset):

    def __init__(self, features, labels=None):

        self.features = features
        self.labels = labels

    def __len__(self):

        return len(self.features)

    def __getitem__(self, index):

        if self.labels is None:

            return self.features[index]

        return self.features[index], self.labels[index]

# =========================================================
# Create Dataset
# =========================================================

train_dataset = RULDataset(
    X_train,
    y_train
)

val_dataset = RULDataset(
    X_val,
    y_val
)

test_dataset = RULDataset(
    X_test
)

# =========================================================
# Create DataLoader
# =========================================================

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================================================
# Verify DataLoader
# =========================================================

print("\n" + "=" * 60)
print("DataLoader Information")
print("=" * 60)

print("Training Batches  :", len(train_loader))
print("Validation Batches:", len(val_loader))
print("Testing Batches   :", len(test_loader))

# =========================================================
# Verify One Batch
# =========================================================

for features, labels in train_loader:

    print("\nFeature Shape :", features.shape)

    print("Label Shape   :", labels.shape)

    break

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("DATASET PIPELINE READY")
    print("=" * 60)

    print("Train Tensor :", X_train.shape)

    print("Validation Tensor :", X_val.shape)

    print("Test Tensor :", X_test.shape)