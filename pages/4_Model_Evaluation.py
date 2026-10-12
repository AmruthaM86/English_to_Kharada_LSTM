import os
import json
import streamlit as st

from utils.helper import load_css


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD CSS
# ============================================================

load_css()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

EVALUATION = os.path.join(
    MODEL_DIR,
    "evaluation.json"
)

BLEU_FILE = os.path.join(
    MODEL_DIR,
    "bleu_score.txt"
)

TRANSLATION_METRICS = os.path.join(
    MODEL_DIR,
    "translation_metrics.txt"
)

LOSS_GRAPH = os.path.join(
    MODEL_DIR,
    "training_loss.png"
)

ACCURACY_GRAPH = os.path.join(
    MODEL_DIR,
    "training_accuracy.png"
)

MODEL_SUMMARY = os.path.join(
    MODEL_DIR,
    "model_summary.txt"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "📊 Model Evaluation"
)

st.write(
    "Evaluation of the Kharada → English "
    "Neural Machine Translation model."
)


# ============================================================
# CHECK EVALUATION FILE
# ============================================================

if not os.path.exists(EVALUATION):

    st.warning(
        "Evaluation results are not available yet."
    )

    st.info(
        "Run train.py first to generate "
        "the evaluation results."
    )

    st.stop()


# ============================================================
# LOAD EVALUATION JSON
# ============================================================

with open(
    EVALUATION,
    "r",
    encoding="utf-8"
) as f:

    evaluation = json.load(f)


# ============================================================
# TEST DATASET TRANSLATION METRICS
# ============================================================

st.header(
    "🧪 Test Dataset Translation Metrics"
)

st.write(
    "BLEU, chrF, and COMET are calculated "
    "using translations generated from the "
    "unseen test dataset."
)


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# BLEU
# ------------------------------------------------------------

with col1:

    bleu = evaluation.get(
        "BLEU Score",
        0
    )

    st.metric(
        "BLEU Score",
        f"{bleu:.4f}"
    )


# ------------------------------------------------------------
# chrF
# ------------------------------------------------------------

with col2:

    chrf = evaluation.get(
        "chrF Score",
        0
    )

    st.metric(
        "chrF Score",
        f"{chrf:.2f}"
    )


# ------------------------------------------------------------
# COMET
# ------------------------------------------------------------

with col3:

    comet = evaluation.get(
        "COMET Score",
        0
    )

    st.metric(
        "COMET Score",
        f"{comet:.4f}"
    )


# ============================================================
# TEST SET MODEL METRICS
# ============================================================

st.header(
    "📈 Test Dataset Model Metrics"
)


col1, col2 = st.columns(2)


with col1:

    test_accuracy = evaluation.get(
        "Test Accuracy",
        0
    )

    st.metric(
        "Test Accuracy",
        f"{test_accuracy:.4f}"
    )


with col2:

    test_loss = evaluation.get(
        "Test Loss",
        0
    )

    st.metric(
        "Test Loss",
        f"{test_loss:.4f}"
    )


# ============================================================
# TRAINING AND VALIDATION METRICS
# ============================================================

st.header(
    "📚 Training and Validation Metrics"
)


col1, col2 = st.columns(2)


with col1:

    training_accuracy = evaluation.get(
        "Final Training Accuracy",
        0
    )

    st.metric(
        "Training Accuracy",
        f"{training_accuracy:.4f}"
    )


with col2:

    validation_accuracy = evaluation.get(
        "Final Validation Accuracy",
        0
    )

    st.metric(
        "Validation Accuracy",
        f"{validation_accuracy:.4f}"
    )


col1, col2 = st.columns(2)


with col1:

    training_loss = evaluation.get(
        "Final Training Loss",
        0
    )

    st.metric(
        "Training Loss",
        f"{training_loss:.4f}"
    )


with col2:

    validation_loss = evaluation.get(
        "Final Validation Loss",
        0
    )

    st.metric(
        "Validation Loss",
        f"{validation_loss:.4f}"
    )


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

st.header(
    "📉 Training and Validation Loss"
)

if os.path.exists(LOSS_GRAPH):

    st.image(
        LOSS_GRAPH,
        caption="Training and Validation Loss"
    )

else:

    st.warning(
        "Training loss graph not found."
    )


# ============================================================
# TRAINING ACCURACY GRAPH
# ============================================================

st.header(
    "📊 Training and Validation Accuracy"
)

if os.path.exists(ACCURACY_GRAPH):

    st.image(
        ACCURACY_GRAPH,
        caption="Training and Validation Accuracy"
    )

else:

    st.warning(
        "Training accuracy graph not found."
    )


# ============================================================
# TRANSLATION METRICS FILE
# ============================================================

st.header(
    "📄 Translation Metrics"
)

if os.path.exists(TRANSLATION_METRICS):

    with open(
        TRANSLATION_METRICS,
        "r",
        encoding="utf-8"
    ) as f:

        metrics_text = f.read()

    st.code(
        metrics_text,
        language="text"
    )

    with open(
        TRANSLATION_METRICS,
        "rb"
    ) as f:

        st.download_button(
            label="⬇️ Download Translation Metrics",
            data=f,
            file_name="translation_metrics.txt",
            mime="text/plain"
        )

else:

    st.info(
        "Translation metrics file has not "
        "been generated yet."
    )


# ============================================================
# MODEL SUMMARY
# ============================================================

st.header(
    "🧠 Model Architecture"
)

if os.path.exists(MODEL_SUMMARY):

    with open(
        MODEL_SUMMARY,
        "r",
        encoding="utf-8"
    ) as f:

        summary = f.read()

    st.code(
        summary,
        language="text"
    )

else:

    st.warning(
        "Model summary not found."
    )


# ============================================================
# COMPLETE EVALUATION JSON
# ============================================================

st.header(
    "📋 Complete Evaluation Results"
)

st.json(
    evaluation
)


# ============================================================
# DOWNLOAD EVALUATION JSON
# ============================================================

with open(
    EVALUATION,
    "rb"
) as f:

    st.download_button(
        label="⬇️ Download Evaluation JSON",
        data=f,
        file_name="evaluation.json",
        mime="application/json"
    )