import streamlit as st

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Kharada → English NMT Project",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Vibrant Soft Gradient Background + High Contrast Text
st.markdown("""
    <style>
    /* Main App Background - Soft Vibrant Gradient */
    .stApp {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 50%, #dbeafe 100%);
        color: #0f172a;
    }
    
    /* Force main headings and standard text to remain sharp and readable */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #0f172a !important;
    }

    /* Main Title Accent Styling */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1d4ed8, #0284c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #334155 !important;
        font-size: 1.25rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism Feature Card Container */
    .content-card {
        background-color: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(191, 219, 254, 0.8);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Vibrant Tag Badge */
    .badge-tag {
        background: linear-gradient(90deg, #2563eb, #0284c7);
        color: #ffffff !important;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    
    .card-heading {
        color: #1e3a8a !important;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .card-text {
        color: #334155 !important;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# Header Section
# ==========================================
st.markdown('<div class="badge-tag">DEEP LEARNING & NATURAL LANGUAGE PROCESSING</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🌍 Kharada → English NMT System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Neural Machine Translation for Low-Resource Languages using Seq2Seq LSTMs</div>', unsafe_allow_html=True)

st.divider()

# ==========================================
# Project Summary Card
# ==========================================
st.markdown("""
<div class="content-card">
    <h2 style="color: #1e3a8a !important; margin-top: 0;">📌 About the Project</h2>
    <p style="color: #334155 !important; font-size: 1.05rem; line-height: 1.7;">
        <b>Kharada → English Translator</b> is an AI-powered Neural Machine Translation system designed specifically to translate 
        Kharada text into English. The project addresses translation challenges in <b>low-resource languages</b> by implementing a 
        <b>Sequence-to-Sequence (Seq2Seq) Encoder–Decoder architecture</b> with <b>Long Short-Term Memory (LSTM)</b> networks.
    </p>
    <p style="color: #334155 !important; font-size: 1.05rem; line-height: 1.7;">
        The <b>Encoder</b> processes the input Kharada sentence and captures its complete contextual vector, while the <b>Decoder</b> 
        generates the corresponding English sentence step-by-step. Built using <b>Python, TensorFlow, and Keras</b>, performance is validated 
        using industry-standard metrics like the <b>BLEU score</b>.
    </p>
</div>
""", unsafe_allow_html=True)

st.space(1)

# ==========================================
# Core Highlights Grid
# ==========================================
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

st.space(2)

# ==========================================
# Workflow Pipeline Breakdown
# ==========================================
st.subheader("🔄 End-to-End Processing Workflow")

col_a, col_b, col_c, col_d, col_e = st.columns(5)

steps = [
    ("1. Kharada Input", "Raw sentence entered by the user."),
    ("2. Preprocessing", "Tokenization, cleaning, and padding sequences."),
    ("3. Encoder (LSTM)", "Compresses text into a contextual vector."),
    ("4. Decoder (LSTM)", "Unrolls sequence to generate English output."),
    ("5. Final Output", "Fluent English translated output sentence.")
]

cols = [col_a, col_b, col_c, col_d, col_e]

for idx, (title, desc) in enumerate(steps):
    with cols[idx]:
        st.markdown(f"""
        <div class="content-card" style="padding: 16px; text-align: center; height: 100%;">
            <div class="card-heading">{title}</div>
            <div class="card-text">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.space(2)

# ==========================================
# Technical Details Section
# ==========================================
with st.expander("🛠️ Comprehensive Technical Specifications"):
    st.markdown("""
    * **Model Architecture:** Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) cells.
    * **Sequence Length Management:** Dynamic padding and mask propagation across variable input length.
    * **Target Domain:** Low-resource dialectal Machine Translation (Kharada dialect to English).
    * **Evaluation Criteria:** Bilingual Evaluation Understudy (BLEU) Score & Validation Cross-Entropy Loss.
    * **Development Stack:** Python 3, TensorFlow 2.x, Keras, NumPy, Streamlit.
    """)

# ==========================================
# Footer
# ==========================================
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #475569; font-size: 0.9rem; font-weight: 500; padding: 10px 0;">
        Kharada → English Neural Machine Translation Project • Built with TensorFlow, Keras & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)