import os
import streamlit as st

from inference import translate
from utils.history import save_history
from utils.helper import load_css

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS if available
try:
    load_css()
except Exception:
    pass

# Custom Styling: Soft Vibrant Gradient Background + Card Layout
st.markdown("""
    <style>
    /* Main App Background - Soft Indigo/Sky Gradient */
    .stApp {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 50%, #dbeafe 100%);
        color: #0f172a;
    }

    /* Ensure strong text contrast */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0f172a !important;
    }

    /* Single Instruction Box Header */
    .instruction-card {
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(191, 219, 254, 0.9);
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a8a !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }

    /* Input & Output Box Container Cards */
    .box-card {
        background-color: #ffffff;
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    /* Output Display Text */
    .output-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e3a8a !important;
        min-height: 120px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# Session State Initialization
# ==========================================================
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "translation" not in st.session_state:
    st.session_state.translation = ""

# ==========================================================
# Single Clean Instruction
# ==========================================================
st.markdown("""
<div class="instruction-card">
    📝 Enter the <b>Kharada</b> sentence in the top box and click <b>Translate</b> to receive the <b>English</b> sentence in the bottom box.
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Box 1: First Language (Kharada Input)
# ==========================================================
st.markdown("### 🗣️ Kharada Language (Input)")

st.text_area(
    label="Kharada Input Sentence",
    key="sentence",
    height=140,
    placeholder="Enter Kharada sentence here...",
    label_visibility="collapsed"
)

st.space(1)

# ==========================================================
# Translator Button in the Middle
# ==========================================================
col_left, col_btn1, col_btn2, col_right = st.columns([2, 2, 1, 2])

with col_btn1:
    translate_clicked = st.button("🌐 Translate", type="primary", use_container_width=True)

# with col_btn2:
#     clear_clicked = st.button("🗑️ Clear", use_container_width=True)

# ==========================================================
# Translation Logic
# ==========================================================
if translate_clicked:
    if st.session_state.sentence.strip() == "":
        st.warning("⚠️ Please enter a Kharada sentence first.")
    else:
        with st.spinner("🤖 Translating..."):
            st.session_state.translation = translate(st.session_state.sentence)
            
            # Save history
            save_history(
                st.session_state.sentence,
                st.session_state.translation
            )

# if clear_clicked:
#     st.session_state.sentence = ""
#     st.session_state.translation = ""
#     st.rerun()

st.space(1)

# ==========================================================
# Box 2: Second Language (English Output)
# ==========================================================
st.markdown(" ### English Language (Output)")

if st.session_state.translation != "":
    st.markdown(
        f'''
        <div class="box-card">
            <div class="output-text">{st.session_state.translation}</div>
        </div>
        ''', 
        unsafe_allow_html=True
    )
    
    st.space(1)
    
    col_a, col_download, col_b = st.columns([3, 2, 3])
    with col_download:
        st.download_button(
            label="⬇️ Download Translation",
            data=st.session_state.translation,
            file_name="translation.txt",
            mime="text/plain",
            use_container_width=True
        )
else:
    st.markdown(
        '''
        <div class="box-card">
            <div class="output-text" style="color: #94a3b8 !important;">
                Translation output will appear here after clicking the Translate button...
            </div>
        </div>
        ''', 
        unsafe_allow_html=True
    )

# ==========================================================
# Footer
# ==========================================================
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #475569; font-size: 0.9rem; font-weight: 600; padding: 10px 0;">
        © 2026 | Kharada → English Translator | TensorFlow • Keras • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)