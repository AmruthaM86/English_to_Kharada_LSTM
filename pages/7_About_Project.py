import streamlit as st
import os
from utils.helper import load_css

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="About Project",
    page_icon="📖",
    layout="wide"
)

load_css()

# ==========================================================
# Asset Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGO = os.path.join(BASE_DIR, "assets", "logo.png")

# ==========================================================
# Header
# ==========================================================

col1, col2 = st.columns([1,6])

with col1:
    if os.path.exists(LOGO):
        st.image(LOGO, width=90)

with col2:
    st.title("📖 About the Project")
    st.write(
        "Kharada → English Neural Machine Translation using Seq2Seq Encoder–Decoder LSTM"
    )

st.divider()

# ==========================================================
# Project Overview
# ==========================================================

st.header("🌍 Project Overview")

st.write("""
This project translates **Kharada language sentences into English**
using a **Neural Machine Translation (NMT)** model.

The application is built using a **Sequence-to-Sequence (Seq2Seq)**
Encoder–Decoder architecture with **Long Short-Term Memory (LSTM)**
networks.

The trained model learns the relationship between Kharada and English
sentences and predicts the corresponding English translation.
""")

st.divider()

# ==========================================================
# Objectives
# ==========================================================

st.header("🎯 Objectives")

st.markdown("""
- Translate Kharada sentences into English.
- Preserve the meaning of the original sentence.
- Demonstrate Neural Machine Translation.
- Build an interactive Streamlit application.
- Evaluate translation quality using BLEU Score.
""")

st.divider()

# ==========================================================
# Technologies
# ==========================================================

st.header("🛠 Technologies Used")

col1, col2 = st.columns(2)

with col1:

    st.success("""
### Machine Learning

- TensorFlow
- Keras
- NumPy
- Joblib
""")

with col2:

    st.info("""
### Application

- Streamlit
- Python
- Matplotlib
- Pandas
""")

st.divider()

# ==========================================================
# Workflow
# ==========================================================

st.header("⚙ Translation Workflow")

st.markdown("""
Kharada Sentence

    │

    ▼

Text Preprocessing

    │

    ▼

Input Tokenization

    │

    ▼

Encoder LSTM

    │

    ▼

Context Vector

    │

    ▼

Decoder LSTM

    │

    ▼

English Sentence

""")

st.divider()

# ==========================================================
# Features
# ==========================================================

st.header("✨ Features")

feature1, feature2, feature3 = st.columns(3)

with feature1:
    st.info("""
### 🌐 Translation

Real-time Kharada to English translation.
""")

with feature2:
    st.success("""
### 📊 Evaluation

BLEU Score and Model Evaluation.
""")

with feature3:
    st.warning("""
### 📜 History

Stores translated sentences.
""")

st.divider()

# ==========================================================
# Dataset
# ==========================================================

st.header("📂 Dataset")

st.write("""
The model was trained using a parallel corpus containing
paired **Kharada–English sentences**.

The dataset was divided into:

- Training Set
- Validation Set
- Test Set

The tokenizer converts each sentence into integer sequences before
feeding them into the neural network.
""")

st.divider()

# ==========================================================
# Model Architecture
# ==========================================================

st.header("🧠 Model Architecture")

st.markdown("""
- Encoder LSTM
- Decoder LSTM
- Embedding Layer
- Context Vector
- Dense Output Layer
- Greedy Decoding
""")

st.divider()

# ==========================================================
# Future Enhancements
# ==========================================================

st.header("🚀 Future Enhancements")

st.markdown("""
- Attention Mechanism
- Transformer Model
- Speech-to-Text Translation
- Text-to-Speech Output
- Mobile Application
- Multilingual Translation
""")

st.divider()

# ==========================================================
# Developer
# ==========================================================

st.header("👩‍💻 Developer")

st.write("""
**Project:** Kharada → English Translator

**Technology:** TensorFlow, Keras, Streamlit

**Architecture:** Seq2Seq Encoder–Decoder LSTM

Developed as an academic Neural Machine Translation project.
""")

st.divider()

st.caption("© 2026 | Kharada → English Translator")