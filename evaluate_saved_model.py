import os
import numpy as np
import tensorflow as tf

from utils.dataset import get_data
from utils.evaluation import generate_translation_metrics


# ============================================================
# STEP 1: LOAD TEST DATASET
# ============================================================

print("=" * 60)
print("LOADING TEST DATASET")
print("=" * 60)

(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    X_test,
    Y_test,
    input_tokenizer,
    target_tokenizer,
    input_vocab_size,
    target_vocab_size,
    max_encoder_length,
    max_decoder_length
) = get_data()


print()
print("Test dataset shape:")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# ============================================================
# STEP 2: LOAD SAVED SEQ2SEQ MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING SAVED SEQ2SEQ MODEL")
print("=" * 60)

seq2seq_path = "models/seq2seq.keras"

if not os.path.exists(seq2seq_path):

    print(
        f"ERROR: Model not found: {seq2seq_path}"
    )

    raise FileNotFoundError(seq2seq_path)


training_model = tf.keras.models.load_model(
    seq2seq_path
)

print("Saved seq2seq model loaded successfully.")


# ============================================================
# STEP 3: LOAD SAVED ENCODER MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING SAVED ENCODER MODEL")
print("=" * 60)

encoder_path = "models/encoder.keras"

if not os.path.exists(encoder_path):

    print(
        f"ERROR: Encoder model not found: {encoder_path}"
    )

    raise FileNotFoundError(encoder_path)


encoder_model = tf.keras.models.load_model(
    encoder_path
)

print("Saved encoder model loaded successfully.")


# ============================================================
# STEP 4: LOAD SAVED DECODER MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING SAVED DECODER MODEL")
print("=" * 60)

decoder_path = "models/decoder.keras"

if not os.path.exists(decoder_path):

    print(
        f"ERROR: Decoder model not found: {decoder_path}"
    )

    raise FileNotFoundError(decoder_path)


decoder_model = tf.keras.models.load_model(
    decoder_path
)

print("Saved decoder model loaded successfully.")


# ============================================================
# STEP 5: EVALUATE TEST DATASET
# ============================================================

print()
print("=" * 60)
print("EVALUATING TEST DATASET")
print("=" * 60)


decoder_input_test = Y_test[:, :-1]

decoder_target_test = Y_test[:, 1:]

decoder_target_test = np.expand_dims(
    decoder_target_test,
    axis=-1
)


loss, accuracy = training_model.evaluate(
    [
        X_test,
        decoder_input_test
    ],
    decoder_target_test,
    verbose=1
)


print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Test Loss     : {loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy:.4f}"
)


# ============================================================
# STEP 6: GENERATE TEST TRANSLATIONS
# ============================================================

print()
print("=" * 60)
print("GENERATING TEST TRANSLATIONS")
print("=" * 60)


bleu, chrf, comet = generate_translation_metrics(
    X_test,
    Y_test,
    encoder_model,
    decoder_model,
    input_tokenizer,
    target_tokenizer,
    max_encoder_length,
    max_decoder_length,
    "models"
)


# ============================================================
# STEP 7: FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL TEST DATASET RESULTS")
print("=" * 60)

print(
    f"Test Loss     : {loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy:.4f}"
)

print(
    f"BLEU Score    : {bleu:.6f}"
)

print(
    f"chrF Score    : {chrf:.6f}"
)

print(
    f"COMET Score   : {comet:.6f}"
)

print("=" * 60)


print()
print("Evaluation completed successfully.")

print()
print("Check the actual translations here:")

print(
    "models\\test_predictions.txt"
)