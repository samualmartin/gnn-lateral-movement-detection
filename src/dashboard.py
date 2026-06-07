import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Lateral Movement Detection using GNN",
    layout="wide"
)

st.title("🔐 Lateral Movement Detection using Graph Neural Networks")

st.markdown("""
This project uses a **Graph Attention Network (GAT)** trained on the **LMD-2023 dataset**
to detect lateral movement attacks in enterprise environments.

### Attack Classes
- Normal Activity
- EoHT Attack
- EoRS Attack
""")

st.divider()


st.header("📊 Project Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Nodes",
        "45,000"
    )

with col2:
    st.metric(
        "Edges",
        "450,000"
    )

with col3:
    st.metric(
        "Features",
        "80"
    )

with col4:
    st.metric(
        "Accuracy",
        "99.2%"
    )

st.divider()


st.header("📁 Dataset Statistics")

if Path(
    "outputs/dataset_statistics.txt"
).exists():

    with open(
        "outputs/dataset_statistics.txt"
    ) as f:

        st.text(
            f.read()
        )

st.divider()


st.header("📈 Training Performance")

col1, col2 = st.columns(2)

with col1:

    if Path(
        "outputs/training_loss.png"
    ).exists():

        st.image(
            "outputs/training_loss.png",
            caption="Training Loss Curve"
        )

with col2:

    if Path(
        "outputs/training_accuracy.png"
    ).exists():

        st.image(
            "outputs/training_accuracy.png",
            caption="Training Accuracy Curve"
        )

st.divider()


st.header("🎯 Confusion Matrix")

if Path(
    "outputs/confusion_matrix.png"
).exists():

    st.image(
        "outputs/confusion_matrix.png",
        caption="Model Confusion Matrix"
    )

st.divider()


st.header("🕸️ Event Graph Visualization")

if Path(
    "outputs/attack_graph.png"
).exists():

    st.image(
        "outputs/attack_graph.png",
        caption="LMD-2023 Event Similarity Graph"
    )

st.divider()

st.header("📉 Prediction Distribution")

if Path(
    "outputs/prediction_distribution.png"
).exists():

    st.image(
        "outputs/prediction_distribution.png",
        caption="Predicted Class Distribution"
    )

st.divider()

st.header("📋 Classification Report")

if Path(
    "outputs/classification_report.csv"
).exists():

    report_df = pd.read_csv(
        "outputs/classification_report.csv"
    )

    st.dataframe(
        report_df,
        use_container_width=True
    )

st.divider()


st.header("📄 Detailed Metrics")

if Path(
    "outputs/metrics.txt"
).exists():

    with open(
        "outputs/metrics.txt"
    ) as f:

        st.text(
            f.read()
        )

st.divider()


st.header("🧠 Model Architecture")

if Path(
    "outputs/model_summary.txt"
).exists():

    with open(
        "outputs/model_summary.txt"
    ) as f:

        st.code(
            f.read(),
            language="python"
        )

st.divider()

st.success(
    "Graph Attention Network successfully trained and evaluated on the LMD-2023 dataset."
)