import json
import os

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)

# ==========================================
# BLEU Score
# ==========================================

def calculate_bleu(reference, prediction):

    smoothie = SmoothingFunction().method1

    return sentence_bleu(

        [reference.split()],

        prediction.split(),

        smoothing_function=smoothie

    )

# ==========================================
# Average BLEU
# ==========================================

def average_bleu(references, predictions):

    scores = []

    for ref, pred in zip(references, predictions):

        scores.append(

            calculate_bleu(ref, pred)

        )

    if len(scores) == 0:

        return 0

    return round(sum(scores) / len(scores), 4)

# ==========================================
# Exact Match Accuracy
# ==========================================

def calculate_accuracy(references, predictions):

    correct = 0

    total = len(references)

    for ref, pred in zip(references, predictions):

        if ref.lower().strip() == pred.lower().strip():

            correct += 1

    if total == 0:

        return 0

    return round((correct / total) * 100, 2)

# ==========================================
# Word Accuracy
# ==========================================

def word_accuracy(reference, prediction):

    ref_words = reference.split()

    pred_words = prediction.split()

    if len(ref_words) == 0:

        return 0

    matches = 0

    for r, p in zip(ref_words, pred_words):

        if r.lower() == p.lower():

            matches += 1

    return round(matches / len(ref_words) * 100, 2)

# ==========================================
# Save Metrics
# ==========================================

def save_metrics(

    accuracy,

    bleu,

    filepath

):

    metrics = {

        "accuracy": accuracy,

        "bleu_score": bleu

    }

    with open(

        filepath,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4

        )

# ==========================================
# Load Metrics
# ==========================================

def load_metrics(filepath):

    if not os.path.exists(filepath):

        return {}

    with open(

        filepath,

        "r",

        encoding="utf-8"

    ) as file:

        return json.load(file)

# ==========================================
# Print Metrics
# ==========================================

def print_metrics(

    accuracy,

    bleu

):

    print("=" * 50)

    print("Evaluation Metrics")

    print("=" * 50)

    print(f"Exact Match Accuracy : {accuracy:.2f}%")

    print(f"Average BLEU Score   : {bleu:.4f}")

    print("=" * 50)