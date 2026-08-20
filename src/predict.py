"""
=========================================================
NASA C-MAPSS Remaining Useful Life Prediction
File        : predict.py
Description : Run inference using the saved CNN-LSTM model
=========================================================
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import RULLSTM
from dataset import X_test_scaled, X_test_seq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUL_PATH = os.path.join(BASE_DIR, "..", "dataset", "processed", "rul_processed.csv")


def load_model(model_path: str, input_size: int = 24):
    model = RULLSTM(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model


def predict(model: torch.nn.Module, inputs: np.ndarray, device: torch.device):
    model.to(device)
    model.eval()

    data = torch.tensor(inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(data)

    return outputs.cpu().numpy()


def load_actual_rul(rul_path: str) -> np.ndarray:
    rul_df = pd.read_csv(rul_path)

    if "RUL" not in rul_df.columns:
        raise ValueError(f"Expected column 'RUL' in {rul_path}")

    return rul_df["RUL"].to_numpy()


def build_output_df(predictions: np.ndarray, test_df: pd.DataFrame, actual_rul: np.ndarray) -> pd.DataFrame:
    engine_ids = test_df["engine_id"].unique()
    last_cycles = test_df.groupby("engine_id")["cycle"].max().reset_index(drop=True)

    if len(actual_rul) != len(engine_ids):
        raise ValueError(
            "Number of actual RUL labels does not match number of test engines."
        )

    output_df = pd.DataFrame(
        {
            "engine_id": engine_ids,
            "cycle": last_cycles,
            "actual_RUL": actual_rul.flatten(),
            "predicted_RUL": predictions.flatten(),
        }
    )

    return output_df


def plot_actual_vs_predicted(output_df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))

    plt.plot(
        output_df["engine_id"],
        output_df["actual_RUL"],
        marker="o",
        linestyle="-",
        label="Actual RUL"
    )

    plt.plot(
        output_df["engine_id"],
        output_df["predicted_RUL"],
        marker="x",
        linestyle="--",
        label="Predicted RUL"
    )

    plt.xlabel("Engine ID")
    plt.ylabel("Remaining Useful Life")
    plt.title("Actual vs Predicted RUL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_rul_distribution(output_df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))

    plt.hist(
        output_df["actual_RUL"],
        bins=30,
        edgecolor="black",
        alpha=0.75,
        label="Actual RUL"
    )

    plt.title("Actual RUL Distribution")
    plt.xlabel("Remaining Useful Life")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using the saved CNN-LSTM model."
    )

    parser.add_argument(
        "--model",
        default=os.path.join("..", "models", "best_model.pth"),
        help="Path to the saved model checkpoint."
    )

    parser.add_argument(
        "--output",
        default=os.path.join("..", "results", "predictions.csv"),
        help="Path to save prediction results."
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    if args.device == "cuda" and device.type != "cuda":
        print("CUDA requested but unavailable, using CPU instead.")

    print("Loading model from:", args.model)
    model = load_model(args.model)

    print("Loading actual RUL labels from:", RUL_PATH)
    actual_rul = load_actual_rul(RUL_PATH)

    print("Running inference on test dataset...")
    predictions = predict(model, X_test_seq, device)

    print("Building prediction results...")
    result_df = build_output_df(predictions, X_test_scaled, actual_rul)

    result_df.to_csv(args.output, index=False)
    print("Predictions saved to:", args.output)

    plot_path = os.path.join(output_dir, "actual_vs_predicted_rul.png")
    print("Saving actual vs predicted RUL plot to:", plot_path)
    plot_actual_vs_predicted(result_df, plot_path)

    distribution_path = os.path.join(output_dir, "actual_rul_distribution.png")
    print("Saving actual RUL distribution plot to:", distribution_path)
    plot_rul_distribution(result_df, distribution_path)

    print("Plots saved successfully.")


if __name__ == "__main__":
    main()
