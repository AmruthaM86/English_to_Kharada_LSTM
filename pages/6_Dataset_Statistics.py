import streamlit as st
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from utils.helper import load_css

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Dataset Statistics",
    page_icon="📂",
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
    "dataset.png"
)

TRAIN = os.path.join(BASE_DIR, "dataset", "train.txt")
VALID = os.path.join(BASE_DIR, "dataset", "validation.txt")
TEST = os.path.join(BASE_DIR, "dataset", "test.txt")

INPUT_TOKENIZER = os.path.join(
    BASE_DIR,
    "tokenizer",
    "input_tokenizer.pkl"
)

TARGET_TOKENIZER = os.path.join(
    BASE_DIR,
    "tokenizer",
    "target_tokenizer.pkl"
)

LENGTH_FILE = os.path.join(
    BASE_DIR,
    "tokenizer",
    "lengths.json"
)

# ==========================================
# Helper
# ==========================================

def count_lines(file):

    if not os.path.exists(file):
        return 0

    with open(file, "r", encoding="utf-8") as f:
        return len(f.readlines())

# ==========================================
# Dataset Counts
# ==========================================

train_count = count_lines(TRAIN)
valid_count = count_lines(VALID)
test_count = count_lines(TEST)

total = train_count + valid_count + test_count

# ==========================================
# Tokenizers
# ==========================================

input_vocab = 0
target_vocab = 0

if os.path.exists(INPUT_TOKENIZER):

    tokenizer = joblib.load(INPUT_TOKENIZER)

    input_vocab = len(tokenizer.word_index)

if os.path.exists(TARGET_TOKENIZER):

    tokenizer = joblib.load(TARGET_TOKENIZER)

    target_vocab = len(tokenizer.word_index)

# ==========================================
# Lengths
# ==========================================

encoder_length = 0
decoder_length = 0

if os.path.exists(LENGTH_FILE):

    with open(LENGTH_FILE) as f:

        data = json.load(f)

        encoder_length = data["max_encoder_length"]

        decoder_length = data["max_decoder_length"]

# ==========================================
# Header
# ==========================================

col1, col2 = st.columns([1,8])

with col1:

    if os.path.exists(ICON):
        st.image(ICON,width=70)

with col2:

    st.title("Dataset Statistics")

st.write("Overview of the dataset used for training the model.")

st.divider()

# ==========================================
# Metrics
# ==========================================

m1,m2,m3,m4 = st.columns(4)

m1.metric("Total", total)

m2.metric("Train", train_count)

m3.metric("Validation", valid_count)

m4.metric("Test", test_count)

st.divider()

# ==========================================
# Vocabulary
# ==========================================

v1,v2 = st.columns(2)

v1.metric(
    "Input Vocabulary",
    input_vocab
)

v2.metric(
    "Target Vocabulary",
    target_vocab
)

st.divider()

# ==========================================
# Sequence Lengths
# ==========================================

l1,l2 = st.columns(2)

l1.metric(
    "Max Encoder Length",
    encoder_length
)

l2.metric(
    "Max Decoder Length",
    decoder_length
)

st.divider()

# ==========================================
# Pie Chart
# ==========================================

st.subheader("Dataset Split")

fig, ax = plt.subplots(figsize=(2,2))

ax.pie(

    [train_count, valid_count, test_count],

    labels=[
        "Train",
        "Validation",
        "Test"
    ],

    autopct="  %1.1f%%",

    startangle=45

)

ax.axis("equal")

st.pyplot(fig)

st.divider()

# ==========================================
# Table
# ==========================================

df = pd.DataFrame({

    "Dataset":[

        "Training",

        "Validation",

        "Testing"

    ],

    "Samples":[

        train_count,

        valid_count,

        test_count

    ]

})

st.subheader("Dataset Summary")

st.dataframe(

    df,

    use_container_width=True

)

st.divider()

st.success("Dataset statistics loaded successfully.")