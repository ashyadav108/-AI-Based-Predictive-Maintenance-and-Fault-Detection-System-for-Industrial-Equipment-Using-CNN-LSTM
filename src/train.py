"""
=========================================================
NASA C-MAPSS Remaining Useful Life Prediction
File        : train.py
Description : Train the CNN-LSTM model
=========================================================
"""

# =========================================================
# Import Libraries
# =========================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import RULLSTM
from dataset import train_loader, val_loader

# =========================================================
# Device Configuration
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", device)
print("=" * 60)

# =========================================================
# Create Models Folder
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# Hyperparameters
# =========================================================

INPUT_SIZE = 24
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 64

# =========================================================
# Initialize Model
# =========================================================

model = RULLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

model = model.to(device)

print("\nModel Loaded Successfully.\n")

print(model)

# =========================================================
# Loss Function
# =========================================================

criterion = nn.MSELoss()

print("\nLoss Function : MSELoss")

# =========================================================
# Optimizer
# =========================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

print("Optimizer : Adam")

# =========================================================
# Print Configuration
# =========================================================

print("\n" + "=" * 60)

print("Training Configuration")

print("=" * 60)

print(f"Epochs        : {NUM_EPOCHS}")
print(f"Batch Size    : {BATCH_SIZE}")
print(f"Learning Rate : {LEARNING_RATE}")
print(f"Hidden Size   : {HIDDEN_SIZE}")
print(f"LSTM Layers   : {NUM_LAYERS} (within CNN-LSTM)")

print("=" * 60)

# =========================================================
# Training Function
# =========================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0

    for features, labels in dataloader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss


# =========================================================
# Validation Function
# =========================================================

def validate(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0.0

    with torch.no_grad():

        for features, labels in dataloader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss


# =========================================================
# Lists for Saving Loss
# =========================================================

train_losses = []

val_losses = []

best_val_loss = float("inf")

patience = 10

counter = 0

print("\nStarting Training...\n")

print("=" * 70)
print(f"{'Epoch':<10}{'Train Loss':<20}{'Validation Loss':<20}")

print("=" * 70)

# =========================================================
# Training Loop
# =========================================================

for epoch in range(NUM_EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss = validate(
        model,
        val_loader,
        criterion,
        device
    )

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        counter = 0

        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, "best_model.pth")
        )

        print("Best model saved.")

    else:

        counter += 1

        print(f"EarlyStopping Counter : {counter}/{patience}")

    if counter >= patience:

        print("Early stopping activated.")

        break

plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")

plt.legend()
plt.grid(True)

plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
plt.show()
plt.close()

print("\nTraining Completed Successfully")

print(f"Best Validation Loss : {best_val_loss:.4f}")