import os
import json
import joblib
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

# =====================================================
# Global Variables (Initially Empty)
# =====================================================

encoder_model = None
decoder_model = None

input_tokenizer = None
target_tokenizer = None

max_encoder_length = None
max_decoder_length = None

reverse_target_word = None

# =====================================================
# Lazy Loading Function
# =====================================================

def load_resources():

    global encoder_model
    global decoder_model
    global input_tokenizer
    global target_tokenizer
    global max_encoder_length
    global max_decoder_length
    global reverse_target_word

    # Already loaded
    if encoder_model is not None:
        return

    # -------------------------
    # Load Models
    # -------------------------

    encoder_model = load_model(
        os.path.join(MODEL_DIR, "encoder.keras"),
        compile=False
    )

    decoder_model = load_model(
        os.path.join(MODEL_DIR, "decoder.keras"),
        compile=False
    )

    # -------------------------
    # Load Tokenizers
    # -------------------------

    input_tokenizer = joblib.load(
        os.path.join(TOKENIZER_DIR, "input_tokenizer.pkl")
    )

    target_tokenizer = joblib.load(
        os.path.join(TOKENIZER_DIR, "target_tokenizer.pkl")
    )

    # -------------------------
    # Load Lengths
    # -------------------------

    with open(
        os.path.join(TOKENIZER_DIR, "lengths.json"),
        "r",
        encoding="utf-8"
    ) as f:

        lengths = json.load(f)

    max_encoder_length = lengths["max_encoder_length"]
    max_decoder_length = lengths["max_decoder_length"]

    # -------------------------
    # Reverse Dictionary
    # -------------------------

    reverse_target_word = {
        index: word
        for word, index in target_tokenizer.word_index.items()
    }

    print("Resources Loaded Successfully")

# =====================================================
# Translation
# =====================================================

def translate(sentence):

    load_resources()

    sentence = sentence.strip()

    if sentence == "":
        return ""

    sequence = input_tokenizer.texts_to_sequences(
        [sentence]
    )

    sequence = pad_sequences(
        sequence,
        maxlen=max_encoder_length,
        padding="post"
    )

    states = encoder_model.predict(
        sequence,
        verbose=0
    )

    start_index = target_tokenizer.word_index.get("<start>")
    end_index = target_tokenizer.word_index.get("<end>")

    if start_index is None or end_index is None:
        return "Tokenizer missing <start> or <end>."

    target_seq = np.array([[start_index]])

    translated_words = []

    for _ in range(max_decoder_length):

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states,
            verbose=0
        )

        predicted_id = np.argmax(
            output_tokens[0, -1, :]
        )

        if predicted_id == 0:
            break

        if predicted_id == end_index:
            break

        predicted_word = reverse_target_word.get(predicted_id)

        if predicted_word is None:
            break

        if predicted_word not in (
            "<start>",
            "<end>"
        ):
            translated_words.append(predicted_word)

        target_seq = np.array([[predicted_id]])

        states = [h, c]

    return " ".join(translated_words)

# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("        Kharada → English Translator")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:

        sentence = input("Kharada : ")

        if sentence.lower() == "exit":
            break

        print("English :", translate(sentence))