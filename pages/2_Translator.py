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
    layout="wide"
)

load_css()

# ==========================================================
# Assets
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ICON_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "icons",
    "translate.png"
)

# ==========================================================
# Session State
# ==========================================================

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "translation" not in st.session_state:
    st.session_state.translation = ""

# ==========================================================
# Header
# ==========================================================

col1, col2 = st.columns([1, 8])

with col1:
    if os.path.exists(ICON_PATH):
        st.image(ICON_PATH, width=80)

with col2:
    st.title("🌐 Kharada → English Translator")
    st.write(
        "Translate Kharada sentences into English using the trained Seq2Seq Encoder–Decoder LSTM model."
    )

st.divider()

# ==========================================================
# Statistics
# ==========================================================

c1, c2, c3 = st.columns(3)

c1.metric("Language", "Kharada → English")
c2.metric("Model", "Seq2Seq LSTM")
c3.metric("Framework", "TensorFlow")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# Input
# ==========================================================

st.subheader("✍ Enter Kharada Sentence")

st.text_area(
    "",
    key="sentence",
    height=180,
    placeholder="Example:\nಹಾವ ಹಂಗ ಒಟ್ಟ"
)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# Buttons
# ==========================================================

col1, col2, col3 = st.columns(3)

translate_clicked = col1.button(
    "🌐 Translate",
    use_container_width=True
)

clear_clicked = col2.button(
    "🗑 Clear",
    use_container_width=True
)

download_placeholder = col3.empty()

# ==========================================================
# Translate
# ==========================================================

if translate_clicked:

    if st.session_state.sentence.strip() == "":

        st.warning("Please enter a Kharada sentence.")

    else:

        with st.spinner("🤖 Translating..."):

            st.session_state.translation = translate(
                st.session_state.sentence
            )

        save_history(
            st.session_state.sentence,
            st.session_state.translation
        )

        st.success("Translation Completed Successfully!")

# ==========================================================
# Clear
# ==========================================================

if clear_clicked:

    st.session_state.sentence = ""
    st.session_state.translation = ""

    st.rerun()

# ==========================================================
# Translation Result
# ==========================================================

if st.session_state.translation != "":

    st.divider()

    st.subheader("✅ English Translation")

    st.code(
        st.session_state.translation,
        language=None
    )

    download_placeholder.download_button(
        label="⬇ Download",
        data=st.session_state.translation,
        file_name="translation.txt",
        mime="text/plain",
        use_container_width=True
    )

# ==========================================================
# About Model
# ==========================================================

st.divider()

st.subheader("🤖 About the Model")

st.info(
    """
**Architecture**

- Seq2Seq Encoder–Decoder
- LSTM Neural Network
- TensorFlow & Keras
- Tokenization
- Greedy Decoding
- Neural Machine Translation (NMT)
"""
)

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "© 2026 | Kharada → English Translator | TensorFlow • Keras • Streamlit"
)