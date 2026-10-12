import os
import streamlit as st
from utils.helper import load_css

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Kharada → English NMT System | Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load helper CSS if available
try:
    load_css()
except Exception:
    pass

# Custom Styling: Soft Vibrant Gradient Background + Sharp Dark Text Contrast
st.markdown("""
    <style>
    /* Main App Background - Soft Indigo/Blue Gradient */
    .stApp {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 50%, #dbeafe 100%);
        color: #0f172a;
    }
    
    /* Ensure all natural typography renders sharply against light background */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0f172a !important;
    }

    /* Hero Section Styling */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1d4ed8, #0284c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }

    .hero-subtitle {
        color: #334155 !important;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Glassmorphism Card Container */
    .content-card {
        background-color: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(191, 219, 254, 0.9);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Top Category Badge */
    .badge-tag {
        background: linear-gradient(90deg, #2563eb, #0284c7);
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }

    /* Workflow Pipeline Pill */
    .workflow-pill {
        background-color: #ffffff;
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        font-weight: 700;
        color: #1e3a8a !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: CONTENT FROM APP.PY (PLACED FIRST)
# ==============================================================================

# 1.1 Header & Title Banner
header_col1, header_col2 = st.columns([1, 5], vertical_alignment="center")

with header_col1:
    logo_path = "frontend/assets/logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=110)
    else:
        st.markdown("<h1 style='font-size: 3.5rem; margin:0;'>🌍</h1>", unsafe_allow_html=True)

with header_col2:
    st.markdown('<div class="badge-tag">DEEP LEARNING & NATURAL LANGUAGE PROCESSING</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🌍 Kharada → English NMT System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Neural Machine Translation for Low-Resource Languages using Seq2Seq LSTMs</div>', unsafe_allow_html=True)

st.divider()

# Optional Banner Image
banner_path = "frontend/assets/banner.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)
    st.space(1)

# 1.2 Main Project Description
st.markdown("""
<div class="content-card">
    <h2 style="color: #1e3a8a !important; margin-top: 0;">📌 About the Project</h2>
    <p style="color: #334155 !important; font-size: 1.05rem; line-height: 1.7;">
        <b>Kharada → English Translator</b> is an AI-powered Neural Machine Translation system designed specifically to translate 
        Kharada text into English. The project addresses translation challenges in <b>low-resource languages</b> by implementing a 
        <b>Sequence-to-Sequence (Seq2Seq) Encoder–Decoder architecture</b> with <b>Long Short-Term Memory (LSTM)</b> networks and an <b>Attention Mechanism</b>.
    </p>
    <p style="color: #334155 !important; font-size: 1.05rem; line-height: 1.7;">
        The <b>Encoder</b> processes the input Kharada sentence and captures its complete contextual vector, while the <b>Decoder</b> 
        generates the corresponding English sentence step-by-step. Built using <b>Python, TensorFlow, and Keras</b>, performance is validated 
        using industry-standard metrics like the <b>BLEU score</b>, making the system a practical demonstration of how deep learning can be applied to Neural Machine Translation for low-resource languages.
    </p>
</div>
""", unsafe_allow_html=True)

st.space(1)

# 1.3 Key Highlights Grid
st.subheader("💡 Key Architecture Highlights")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.container(border=True).markdown("""
    ### 🤖 Architecture
    **Seq2Seq LSTM**
    
    Encoder-Decoder model with recurrent memory units for sequence handling.
    """)

with m2:
    st.container(border=True).markdown("""
    ### ⚡ Framework
    **TensorFlow / Keras**
    
    Built and trained using deep learning pipelines in Python.
    """)

with m3:
    st.container(border=True).markdown("""
    ### 📊 Evaluation
    **BLEU Score**
    
    Assessed against ground-truth translations for sentence-level accuracy.
    """)

with m4:
    st.container(border=True).markdown("""
    ### 🌐 Focus Area
    **Low-Resource NMT**
    
    Tackles limited parallel corpus availability for local dialects.
    """)

st.space(1)

# 1.4 End-to-End Workflow Pipeline
st.subheader("🔄 Machine Translation Processing Pipeline")

pipeline_steps = [
    "1. Dataset",
    "2. Preprocessing",
    "3. Tokenizer",
    "4. Encoder (LSTM)",
    "5. Attention",
    "6. Decoder (LSTM)",
    "7. English Output"
]

wf_cols = st.columns(len(pipeline_steps))
for idx, step_name in enumerate(pipeline_steps):
    with wf_cols[idx]:
        st.markdown(f'<div class="workflow-pill">{step_name}</div>', unsafe_allow_html=True)

st.divider()

# ==============================================================================
# SECTION 2: NON-REPEATED CONTENT FROM HOME.PY (PLACED SECOND)
# ==============================================================================

# 2.1 Project Statistics & Features
col_stats, col_features = st.columns([2, 3], gap="medium")

with col_stats:
    st.subheader("📊 Parallel Corpus Split")
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    s1, s2 = st.columns(2)
    s3, s4 = st.columns(2)
    
    with s1:
        st.metric(label="Total Dataset", value="952 Pairs")
    with s2:
        st.metric(label="Training Set", value="761")
    with s3:
        st.metric(label="Validation Set", value="95")
    with s4:
        st.metric(label="Testing Set", value="96")
        
    st.markdown('</div>', unsafe_allow_html=True)

with col_features:
    st.subheader("🚀 Platform Features")
    
    f1, f2 = st.columns(2)
    with f1:
        st.info("""
        ✔ Kharada → English NMT  
        ✔ Deep Learning LSTM  
        ✔ Attention Mechanism  
        ✔ BLEU Score Metric  
        """)
    with f2:
        st.success("""
        ✔ Training Loss & Accuracy Graphs  
        ✔ Translation History Logging  
        ✔ Dataset & Token Statistics  
        ✔ User Authentication System  
        """)

st.space(1)

# 2.2 Hyperparameters Table & Technologies Used
tech_col, nav_col = st.columns([1, 1], gap="medium")

with tech_col:
    st.subheader("🤖 Model Hyperparameters & Tech Stack")
    
    model_specs = {
        "Parameter": ["Model Architecture", "Embedding Dimension", "LSTM Units", "Epochs Trained", "Optimizer", "Loss Function"],
        "Value": ["Seq2Seq + Attention", "256", "512", "50", "Adam", "Sparse Categorical Crossentropy"]
    }
    st.dataframe(model_specs, use_container_width=True, hide_index=True)
    
    st.markdown("**Technologies Used:**")
    t1, t2, t3, t4 = st.columns(4)
    t1.info("Python")
    t2.info("TensorFlow")
    t3.info("Keras")
    t4.info("Streamlit")

with nav_col:
    st.subheader("📂 Application Navigation Guide")
    st.markdown("""
    Use the **left sidebar** to switch between modules:
    
    * **🏠 Home** — Overview, system architecture, and dataset split
    * **🌐 Translator** — Interactive Kharada-to-English prediction engine
    * **📜 History** — Saved translation logs and past inputs
    * **📈 Model Evaluation** — BLEU score benchmarking & error matrix
    * **📊 Training Graphs** — Loss and accuracy curves across epochs
    * **📁 Dataset Statistics** — Vocabulary frequency and token distributions
    * **ℹ️ About Project** — Detailed research & methodology notes
    * **🔐 Login** — User access & account management
    """)

# ==========================================
# Footer
# ==========================================
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #475569; font-size: 0.9rem; font-weight: 600; padding: 10px 0;">
        Developed using ❤️ TensorFlow • Keras • Streamlit | Kharada Neural Machine Translation System
    </div>
    """,
    unsafe_allow_html=True
)