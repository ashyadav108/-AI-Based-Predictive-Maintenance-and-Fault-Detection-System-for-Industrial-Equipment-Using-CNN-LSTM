# Predictive Maintenance using CNN-LSTM

This repository implements a Remaining Useful Life (RUL) prediction pipeline for the NASA C-MAPSS engine dataset. It includes preprocessing, a CNN-LSTM model training pipeline, inference, and a Streamlit dashboard for visualizing predictions.

## Repository Structure

- `app.py` — Streamlit application for viewing predictions and engine health.
- `src/`
  - `model.py` — CNN-LSTM model definition.
  - `dataset.py` — loads processed data, applies scaling, creates train/validation/test sequences, and builds PyTorch DataLoaders.
  - `train.py` — trains the model, saves the best checkpoint, and writes training loss plots.
  - `preprocess.py` — loads raw NASA C-MAPSS files, computes RUL, saves processed CSV files, and generates exploratory charts.
  - `predict.py` — loads the trained model, runs inference on test engines, saves prediction results, and exports comparison plots.
- `dataset/`
  - `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` — raw dataset files.
  - `processed/` — contains processed `train_processed.csv`, `test_processed.csv`, and `rul_processed.csv` after preprocessing.
- `models/`
  - `best_model.pth` — trained model checkpoint saved by `src/train.py`.
- `results/`
  - prediction CSV and chart outputs produced by `src/predict.py`, `src/preprocess.py`, and `src/train.py`.

## Requirements

Install the Python dependencies before running any scripts.

```bash
pip install numpy pandas matplotlib scikit-learn torch streamlit
```

> If you use GPU acceleration, install the appropriate `torch` package for your CUDA version.

## Usage

### 1. Preprocess the data

Generate processed CSV files and exploratory charts.

```bash
python src/preprocess.py
```

This creates:
- `dataset/processed/train_processed.csv`
- `dataset/processed/test_processed.csv`
- `dataset/processed/rul_processed.csv`
- `results/engine_life_distribution.png`
- `results/rul_distribution.png`
- `results/correlation_heatmap.png`

### 2. Train the model

Train the CNN-LSTM model and save the best checkpoint.

```bash
python src/train.py
```

This creates:
- `models/best_model.pth`
- `results/loss_curve.png`

### 3. Run inference

Generate predictions and charts for actual vs predicted RUL.

```bash
python src/predict.py
```

This creates:
- `results/predictions.csv`
- `results/actual_vs_predicted_rul.png`
- `results/actual_rul_distribution.png`

### 4. Launch the dashboard

Open the Streamlit app to inspect engine predictions.

```bash
streamlit run app.py
```

The dashboard uses:
- `models/best_model.pth`
- `results/predictions.csv`

## Workflow

1. `python src/preprocess.py`
   - load raw `dataset/*.txt`
   - compute RUL labels
   - save processed CSV files in `dataset/processed/`
   - generate exploratory charts in `results/`
2. `python src/train.py`
   - load processed train sequences from `dataset/processed/`
   - train the CNN-LSTM model
   - save best weights to `models/best_model.pth`
   - save training loss plot to `results/loss_curve.png`
3. `python src/predict.py`
   - load the trained model checkpoint
   - run inference on processed test sequences
   - save predictions and comparison charts to `results/`
4. `streamlit run app.py`
   - load the trained model and prediction CSV
   - show actual vs predicted RUL and engine health status

## What each script does

- `src/preprocess.py`
  - loads raw C-MAPSS data
  - computes `RUL` for each training observation
  - saves processed datasets to `dataset/processed/`
  - generates exploratory PNG charts

- `src/dataset.py`
  - loads processed train/test CSV files
  - scales features with `MinMaxScaler`
  - builds sequence windows for training and test inputs
  - creates PyTorch datasets and dataloaders

- `src/train.py`
  - defines training configuration and optimizer
  - trains the CNN-LSTM model for up to 50 epochs
  - saves the best model checkpoint
  - plots training and validation loss

- `src/predict.py`
  - loads the saved CNN-LSTM model checkpoint
  - reads processed test sequences
  - loads actual `RUL` values from `dataset/processed/rul_processed.csv`
  - saves prediction results and comparison plots

- `app.py`
  - loads the trained model and `results/predictions.csv`
  - displays per-engine actual vs predicted values in Streamlit
  - shows health status and model metrics

## Notes

- The current Streamlit app expects `results/predictions.csv` and `models/best_model.pth` to exist.
- If the app fails to launch, verify those files exist and run the preprocessing, training, and prediction steps first.
- `src/dataset.py` imports `train_loader`, `val_loader`, and `test_loader` at import time, so training begins only after dataset preparation.

## Troubleshooting

- If `streamlit` is not found, install it in the same Python environment used to run the project.
- If `best_model.pth` is missing, rerun `python src/train.py`.
- If `results/predictions.csv` is missing, rerun `python src/predict.py`.

## Recommended order

1. `python src/preprocess.py`
2. `python src/train.py`
3. `python src/predict.py`
4. `streamlit run app.py`
