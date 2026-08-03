import streamlit as st
import os
from utils.helper import load_css

# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="Kharada Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGO = os.path.join(BASE_DIR, "assets", "logo.png")
BANNER = os.path.join(BASE_DIR, "assets", "banner.png")

# ==========================================
# Hero Section
# ==========================================

col1, col2 = st.columns([1,3])

with col1:

    if os.path.exists(LOGO):
        st.image(LOGO, width=180)

with col2:

    st.markdown("""
# 🌍 Kharada → English Translator

### AI Powered Neural Machine Translation

Translate Kharada sentences into fluent English using a **Seq2Seq Encoder–Decoder LSTM** model built with **TensorFlow** and **Keras**.

""")

st.markdown("---")

# ==========================================
# Banner
# ==========================================

if os.path.exists(BANNER):
    st.image(BANNER, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# Feature Cards
# ==========================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown("""
<div class="feature-card">

## 🌐

### Translation

Real-time AI translation

</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="feature-card">

## ⚡

### Fast

Instant prediction

</div>
""", unsafe_allow_html=True)

with c3:
    st.markdown("""
<div class="feature-card">

## 📊

### Evaluation

BLEU Score & Accuracy

</div>
""", unsafe_allow_html=True)

with c4:
    st.markdown("""
<div class="feature-card">

## 🤖

### Deep Learning

Encoder Decoder LSTM

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# Workflow
# ==========================================

st.markdown("""
<div class="glass">

## 🔄 Translation Workflow

Kharada Sentence

⬇

Preprocessing

⬇

Encoder (LSTM)

⬇

Decoder (LSTM)

⬇

English Translation

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# Footer
# ==========================================

st.caption("Developed using TensorFlow • Keras • Streamlit")