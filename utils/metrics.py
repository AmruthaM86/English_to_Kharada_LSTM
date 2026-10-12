import json
import os

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)

# ==========================================================
# BLEU Score for One Sentence
# ==========================================================

def calculate_bleu(reference, prediction):

    smoothie = SmoothingFunction().method1

    return sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=smoothie
    )


# ==========================================================
# Average BLEU Score
# ==========================================================

def average_bleu(references, predictions):

    scores = []

    for ref, pred in zip(references, predictions):

        score = calculate_bleu(ref, pred)

        scores.append(score)

    if len(scores) == 0:
        return 0.0

    return round(sum(scores) / len(scores), 4)


# ==========================================================
# Exact Match Accuracy
# ==========================================================

def calculate_accuracy(references, predictions):

    correct = 0

    total = len(references)

    for ref, pred in zip(references, predictions):

        if ref.strip().lower() == pred.strip().lower():
            correct += 1

    if total == 0:
        return 0.0

    return round((correct / total) * 100, 2)


# ==========================================================
# Word Accuracy
# ==========================================================

def word_accuracy(reference, prediction):

    ref_words = reference.split()

    pred_words = prediction.split()

    if len(ref_words) == 0:
        return 0.0

    matches = 0

    for r, p in zip(ref_words, pred_words):

        if r.lower() == p.lower():
            matches += 1

    return round((matches / len(ref_words)) * 100, 2)


# ==========================================================
# Average Word Accuracy
# ==========================================================

def average_word_accuracy(references, predictions):

    scores = []

    for ref, pred in zip(references, predictions):

        scores.append(
            word_accuracy(ref, pred)
        )

    if len(scores) == 0:
        return 0.0

    return round(sum(scores) / len(scores), 2)


# ==========================================================
# Save Metrics
# ==========================================================

def save_metrics(

    accuracy,
    bleu,
    training_accuracy,
    validation_accuracy,
    training_loss,
    validation_loss,
    model_dir

):

    metrics = {

        "BLEU Score": float(bleu),

        "Exact Match Accuracy": float(accuracy),

        "Final Training Accuracy": float(training_accuracy),

        "Final Validation Accuracy": float(validation_accuracy),

        "Final Training Loss": float(training_loss),

        "Final Validation Loss": float(validation_loss)

    }

    os.makedirs(model_dir, exist_ok=True)

    with open(

        os.path.join(model_dir, "evaluation.json"),

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4

        )

    with open(

        os.path.join(model_dir, "bleu_score.txt"),

        "w",

        encoding="utf-8"

    ) as file:

        file.write(f"{bleu:.4f}")


# ==========================================================
# Load Metrics
# ==========================================================

def load_metrics(model_dir):

    path = os.path.join(

        model_dir,

        "evaluation.json"

    )

    if not os.path.exists(path):

        return {}

    with open(

        path,

        "r",

        encoding="utf-8"

    ) as file:

        return json.load(file)


# ==========================================================
# Print Metrics
# ==========================================================

def print_metrics(

    accuracy,
    bleu,
    training_accuracy,
    validation_accuracy,
    training_loss,
    validation_loss

):

    print("=" * 60)

    print("Model Evaluation")

    print("=" * 60)

    print(f"Exact Match Accuracy   : {accuracy:.2f}%")

    print(f"Average BLEU Score     : {bleu:.4f}")

    print(f"Training Accuracy      : {training_accuracy:.4f}")

    print(f"Validation Accuracy    : {validation_accuracy:.4f}")

    print(f"Training Loss          : {training_loss:.4f}")

    print(f"Validation Loss        : {validation_loss:.4f}")

    print("=" * 60)