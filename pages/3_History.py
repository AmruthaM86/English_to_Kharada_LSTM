import streamlit as st
import pandas as pd
import os
from utils.helper import load_css

from utils.history import (
    load_history,
    clear_history,
    total_translations
)

# ==========================================
# Page Config
# ==========================================

st.set_page_config(

    page_title="Translation History",

    page_icon="🕒",

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

    "history.png"

)

HISTORY_FILE = os.path.join(

    BASE_DIR,

    "history",

    "translation_history.csv"

)

# ==========================================
# Header
# ==========================================

col1, col2 = st.columns([1,8])

with col1:

    if os.path.exists(ICON):

        st.image(ICON, width=70)

with col2:

    st.title("Translation History")

st.write("View all previously translated sentences.")

st.divider()

# ==========================================
# Statistics
# ==========================================

metric1, metric2 = st.columns(2)

metric1.metric(

    "Total Translations",

    total_translations()

)

if os.path.exists(HISTORY_FILE):

    metric2.metric(

        "History File",

        "Available"

    )

else:

    metric2.metric(

        "History File",

        "Not Found"

    )

st.divider()

# ==========================================
# Load History
# ==========================================

history = load_history()

# ==========================================
# Search
# ==========================================

search = st.text_input(

    "🔍 Search Translation",

    placeholder="Search Kharada or English..."

)

if search:

    history = history[

        history.apply(

            lambda row:

            row.astype(str)

               .str.contains(

                    search,

                    case=False

                ).any(),

            axis=1

        )

    ]

# ==========================================
# Display Table
# ==========================================

st.dataframe(

    history,

    use_container_width=True,

    height=450

)

st.divider()

# ==========================================
# Buttons
# ==========================================

col1, col2 = st.columns(2)

with col1:

    if st.button(

        "🗑 Clear History",

        use_container_width=True

    ):

        clear_history()

        st.success("History Cleared Successfully!")

        st.rerun()

with col2:

    if os.path.exists(HISTORY_FILE):

        with open(

            HISTORY_FILE,

            "rb"

        ) as file:

            st.download_button(

                "⬇ Download History",

                data=file,

                file_name="translation_history.csv",

                mime="text/csv",

                use_container_width=True

            )

st.divider()

st.caption("Translation history is stored locally in translation_history.csv")