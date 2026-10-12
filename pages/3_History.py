import os
import pandas as pd
import streamlit as st
from utils.helper import load_css

from utils.history import (
    clear_history,
    load_history,
    total_translations,
)

# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="Translation History",
    page_icon="🕒",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    load_css()
except Exception:
    pass

# Custom Styling: Matching Soft Vibrant Gradient Background & High-Contrast Cards
st.markdown(
    """
    <style>
    /* Main App Background - Soft Indigo/Sky Gradient */
    .stApp {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 50%, #dbeafe 100%);
        color: #0f172a;
    }

    /* Ensure text renders with strong contrast and legibility */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0f172a !important;
    }

    /* Top Instruction / Info Card */
    .info-card {
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(191, 219, 254, 0.9);
        border-radius: 12px;
        padding: 16px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Stat Cards Container */
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ICON = os.path.join(BASE_DIR, "assets", "icons", "history.png")

HISTORY_FILE = os.path.join(
    BASE_DIR, "history", "translation_history.csv"
)

# ==========================================
# Header
# ==========================================

col1, col2 = st.columns([1, 8], vertical_alignment="center")

with col1:
    if os.path.exists(ICON):
        st.image(ICON, width=70)
    else:
        st.markdown(
            "<h1 style='font-size: 2.8rem; margin:0;'>🕒</h1>",
            unsafe_allow_html=True,
        )

with col2:
    st.title("Translation History")

st.markdown(
    """
<div class="info-card">
    🔍 View, search, export, or clear all previously translated sentences from your local session records.
</div>
""",
    unsafe_allow_html=True,
)

# ==========================================
# Statistics
# ==========================================

metric1, metric2 = st.columns(2)

with metric1:
    st.metric("Total Translations", total_translations())

with metric2:
    if os.path.exists(HISTORY_FILE):
        st.metric("History File Status", "Available")
    else:
        st.metric("History File Status", "Not Found")

st.divider()

# ==========================================
# Load History
# ==========================================

history = load_history()

# ==========================================
# Search
# ==========================================

search = st.text_input(
    "🔍 Search Translation", placeholder="Search Kharada or English phrase..."
)

if search:
    history = history[
        history.apply(
            lambda row: row.astype(str)
            .str.contains(search, case=False)
            .any(),
            axis=1,
        )
    ]

# ==========================================
# Display Table
# ==========================================

st.dataframe(history, use_container_width=True, height=450)

st.divider()

# ==========================================
# Buttons
# ==========================================

col1, col2 = st.columns(2)

with col1:
    if st.button("🗑 Clear History", use_container_width=True):
        clear_history()
        st.success("History Cleared Successfully!")
        st.rerun()

with col2:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "rb") as file:
            st.download_button(
                "⬇ Download History",
                data=file,
                file_name="translation_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.divider()

st.caption(
    "© 2026 | Translation history is stored locally in translation_history.csv"
)