"""
=========================================================
NASA C-MAPSS RUL Prediction
File : model.py
Purpose : CNN-LSTM Model Definition
=========================================================
"""

# =====================================================
# Import Libraries
# =====================================================

import torch
import torch.nn as nn

# =====================================================
# CNN-LSTM Model
# =====================================================

class RULLSTM(nn.Module):

    def __init__(
        self,
        input_size=24,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    ):

        super(RULLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1D CNN layers for temporal feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    # =================================================
    # Forward Pass
    # =================================================

    def forward(self, x):

        # Conv1D expects input shaped (batch, channels, seq_len)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Back to (batch, seq_len, channels) for LSTM
        x = x.transpose(1, 2)

        output, (hidden, cell) = self.lstm(x)

        # Last Time Step
        x = output[:, -1, :]

        # Dense Layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Final Prediction
        x = self.fc2(x)

        return x.squeeze()

# =====================================================
# Test Model
# =====================================================

if __name__ == "__main__":

    model = RULLSTM()

    print(model)

    dummy_input = torch.randn(64, 30, 24)

    output = model(dummy_input)

    print("\nInput Shape :", dummy_input.shape)

    print("Output Shape:", output.shape)