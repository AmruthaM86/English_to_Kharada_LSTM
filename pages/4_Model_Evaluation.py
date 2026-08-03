import streamlit as st
import json
import os
import pandas as pd
from utils.helper import load_css

# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide"
)
load_css()
# ==========================================
# Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ICON = os.path.join(
    BASE_DIR,
    "assets",
    "icons",
    "evaluation.png"
)

EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "models",
    "evaluation.json"
)

BLEU_FILE = os.path.join(
    BASE_DIR,
    "models",
    "bleu_score.txt"
)

CONFUSION = os.path.join(
    BASE_DIR,
    "models",
    "confusion_matrix.png"
)

# ==========================================
# Header
# ==========================================

col1, col2 = st.columns([1,8])

with col1:

    if os.path.exists(ICON):
        st.image(ICON, width=70)

with col2:

    st.title("Model Evaluation")

st.write(
    "Performance of the trained Seq2Seq Encoder-Decoder LSTM model."
)

st.divider()

# ==========================================
# Read Evaluation File
# ==========================================

accuracy = "N/A"
bleu = "N/A"

if os.path.exists(EVALUATION_FILE):

    with open(EVALUATION_FILE,"r") as f:

        data = json.load(f)

        accuracy = data.get("accuracy","N/A")
        bleu = data.get("bleu_score","N/A")

# ==========================================
# Metrics
# ==========================================

m1,m2,m3,m4 = st.columns(4)

m1.metric(
    "Accuracy",
    accuracy
)

m2.metric(
    "BLEU Score",
    bleu
)

m3.metric(
    "Model",
    "Seq2Seq"
)

m4.metric(
    "Architecture",
    "LSTM"
)

st.divider()

# ==========================================
# BLEU Score
# ==========================================

st.subheader("BLEU Score")

if os.path.exists(BLEU_FILE):

    with open(BLEU_FILE,"r") as file:

        st.code(file.read())

else:

    st.info("BLEU Score file not found.")

st.divider()

# ==========================================
# Confusion Matrix
# ==========================================

st.subheader("Confusion Matrix")

if os.path.exists(CONFUSION):

    st.image(
        CONFUSION,
        use_container_width=True
    )

else:

    st.info("Confusion Matrix not generated.")

st.divider()

# ==========================================
# Evaluation JSON
# ==========================================

st.subheader("Evaluation Report")

if os.path.exists(EVALUATION_FILE):

    st.json(data)

else:

    st.warning("evaluation.json not found.")

st.divider()

# ==========================================
# Download Report
# ==========================================

if os.path.exists(EVALUATION_FILE):

    with open(EVALUATION_FILE,"rb") as file:

        st.download_button(

            "⬇ Download Evaluation Report",

            data=file,

            file_name="evaluation.json",

            mime="application/json"

        )

st.divider()

st.success("Model Evaluation Completed Successfully.")