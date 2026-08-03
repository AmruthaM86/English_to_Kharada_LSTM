import streamlit as st
import os
from utils.helper import load_css
# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

# ==========================================
# Custom CSS
# ==========================================
load_css()
st.markdown("""
<style>

.main-title{
    font-size:45px;
    font-weight:bold;
    color:#0F62FE;
}

.sub-title{
    font-size:20px;
    color:gray;
}

.card{
    background-color:#F5F7FA;
    padding:20px;
    border-radius:15px;
    border:1px solid #D9D9D9;
    box-shadow:2px 2px 10px rgba(0,0,0,0.08);
}

.metric{
    text-align:center;
    font-size:30px;
    font-weight:bold;
    color:#0F62FE;
}

.footer{
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Header
# ==========================================

col1,col2=st.columns([1,4])

with col1:

    logo="frontend/assets/logo.png"

    if os.path.exists(logo):
        st.image(logo,width=120)

with col2:

    st.markdown(
        "<div class='main-title'>Kharada → English Translator</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>Seq2Seq Encoder-Decoder LSTM with Attention</div>",
        unsafe_allow_html=True
    )

st.divider()

# ==========================================
# Banner
# ==========================================

banner="frontend/assets/banner.png"

if os.path.exists(banner):
    st.image(banner,use_container_width=True)

# ==========================================
# Introduction
# ==========================================

st.markdown("## 📖 Project Overview")

st.write("""
This project translates **Kharada language** into **English**
using a deep learning **Seq2Seq Encoder-Decoder LSTM** architecture.

The application provides:

- Translation
- Model Evaluation
- Dataset Statistics
- Training Graphs
- Translation History
- Interactive Dashboard

""")

st.divider()

# ==========================================
# Statistics
# ==========================================

st.markdown("## 📊 Project Statistics")

c1,c2,c3,c4=st.columns(4)

with c1:
    st.metric("Dataset","952")

with c2:
    st.metric("Training","761")

with c3:
    st.metric("Validation","95")

with c4:
    st.metric("Testing","96")

st.divider()

# ==========================================
# Features
# ==========================================

st.markdown("## 🚀 Features")

c1,c2=st.columns(2)

with c1:

    st.info("""
✔ Kharada → English Translation

✔ Deep Learning LSTM

✔ Attention Mechanism

✔ BLEU Score

✔ Translation History

✔ Login System

""")

with c2:

    st.success("""
✔ Training Graphs

✔ Dataset Statistics

✔ Confusion Matrix

✔ Accuracy

✔ Loss Curve

✔ Streamlit Interface

""")

st.divider()

# ==========================================
# Workflow
# ==========================================

st.markdown("## 🔄 Workflow")

st.image(
"https://raw.githubusercontent.com/dair-ai/ML-YouTube-Courses/main/images/seq2seq.png",
use_container_width=True
)

st.write("""

Dataset

⬇

Preprocessing

⬇

Tokenizer

⬇

Encoder

⬇

Decoder

⬇

Attention

⬇

Translation

""")

st.divider()

# ==========================================
# Technologies
# ==========================================

st.markdown("## 💻 Technologies Used")

tech1,tech2,tech3,tech4=st.columns(4)

with tech1:
    st.success("Python")

with tech2:
    st.success("TensorFlow")

with tech3:
    st.success("Keras")

with tech4:
    st.success("Streamlit")

st.divider()

# ==========================================
# Model Information
# ==========================================

st.markdown("## 🤖 Model Information")

st.table({

"Parameter":[
"Model",
"Embedding",
"LSTM Units",
"Epochs",
"Optimizer",
"Loss Function"
],

"Value":[
"Seq2Seq",
"256",
"512",
"50",
"Adam",
"Sparse Categorical Crossentropy"
]

})

st.divider()

# ==========================================
# Navigation
# ==========================================

st.markdown("## 📂 Navigate")

st.write("""
Use the left sidebar to open:

🏠 Home

🌐 Translator

📜 History

📈 Model Evaluation

📊 Training Graphs

📁 Dataset Statistics

ℹ About Project

🔐 Login
""")

st.divider()

# ==========================================
# Footer
# ==========================================

st.markdown(
"""
<div class='footer'>

Developed using ❤️ TensorFlow • Keras • Streamlit

</div>
""",
unsafe_allow_html=True
)