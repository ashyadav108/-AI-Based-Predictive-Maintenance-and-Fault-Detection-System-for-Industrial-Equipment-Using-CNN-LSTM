import streamlit as st
import torch
import numpy as np
import pandas as pd
import os

from src.model import RULLSTM

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("Predictive Maintenance using CNN-LSTM")
st.markdown("---")

device = torch.device("cpu")

model = RULLSTM(
    input_size=24,
    hidden_size=128,
    num_layers=2,
    dropout=0.3
)

MODEL_PATH = os.path.join(
    "models",
    "best_model.pth"
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

st.success("Model Loaded Successfully")

results = pd.read_csv(
    "results/predictions.csv"
)

st.subheader("Prediction Results")

st.dataframe(results.head())

engine = st.selectbox(
    "Select Engine",
    results["engine_id"].unique()
)

engine_data = results[
    results["engine_id"] == engine
]

st.metric(
    "Predicted RUL",
    f"{engine_data['predicted_RUL'].iloc[0]:.2f}"
)

st.metric(
    "Actual RUL",
    int(engine_data["actual_RUL"].iloc[0])
)

prediction = engine_data["predicted_RUL"].iloc[0]

if prediction > 100:
    st.success("Healthy Engine")

elif prediction > 50:
    st.warning("Maintenance Required Soon")

else:
    st.error("Immediate Maintenance Required")

    st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "10.83")

with col2:
    st.metric("RMSE", "14.63")

with col3:
    st.metric("R² Score", "0.876")

    st.subheader("Prediction Comparison")

st.line_chart(
    engine_data[
        ["actual_RUL", "predicted_RUL"]
    ]
)

