import os
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint

from utils.dataset import get_data
from model import build_models

from utils.evaluation import (
    save_training_graphs,
    save_model_summary,
    save_evaluation_json,
    generate_translation_metrics
)


# ==========================================================
# Create Folders
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)


# ==========================================================
# Load Dataset
# ==========================================================

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


# Print dataset shapes
print("\n===================================")
print("Dataset Shapes")
print("===================================")

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_valid shape:", X_valid.shape)
print("Y_valid shape:", Y_valid.shape)

print("X_test shape :", X_test.shape)
print("Y_test shape :", Y_test.shape)


# ==========================================================
# Teacher Forcing
# ==========================================================

decoder_input_train = Y_train[:, :-1]

decoder_target_train = Y_train[:, 1:]


decoder_input_valid = Y_valid[:, :-1]

decoder_target_valid = Y_valid[:, 1:]


# Keras sparse categorical crossentropy
# expects target shape: (samples, sequence_length, 1)

decoder_target_train = np.expand_dims(
    decoder_target_train,
    axis=-1
)

decoder_target_valid = np.expand_dims(
    decoder_target_valid,
    axis=-1
)


# ==========================================================
# Build Models
# ==========================================================

training_model, encoder_model, decoder_model = build_models(

    input_vocab_size=input_vocab_size,

    target_vocab_size=target_vocab_size,

    embedding_dim=256,

    latent_dim=512

)


print("\n===================================")
print("Model Summary")
print("===================================")

training_model.summary()


# ==========================================================
# Callback - Model Checkpoint
# ==========================================================

checkpoint = ModelCheckpoint(

    filepath="models/checkpoints/best_model.keras",

    monitor="val_loss",

    save_best_only=True,

    save_weights_only=False,

    verbose=1

)


# ==========================================================
# Train Model
# ==========================================================

print("\n===================================")
print("Starting Training")
print("===================================\n")


history = training_model.fit(

    x=[
        X_train,
        decoder_input_train
    ],

    y=decoder_target_train,

    validation_data=(

        [
            X_valid,
            decoder_input_valid
        ],

        decoder_target_valid

    ),

    epochs=30,

    batch_size=32,

    callbacks=[
        checkpoint
    ]

)


# ==========================================================
# Save Models
# ==========================================================

training_model.save(
    "models/seq2seq.keras"
)

encoder_model.save(
    "models/encoder.keras"
)

decoder_model.save(
    "models/decoder.keras"
)


print("\n===================================")
print("Models Saved Successfully")
print("===================================")


# ==========================================================
# Evaluate Model on TEST Dataset
# ==========================================================

print("\n===================================")
print("Evaluating TEST Dataset")
print("===================================\n")


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


print("\n===================================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print("===================================")


# ==========================================================
# Save Training Graphs
# ==========================================================

print("\nStep 1: Saving training graphs...")

save_training_graphs(

    history,

    "models"

)

print("Training graphs saved.")


# ==========================================================
# Save Model Summary
# ==========================================================

print("\nStep 2: Saving model summary...")

save_model_summary(

    training_model,

    "models"

)

print("Model summary saved.")


# ==========================================================
# Generate TEST Translation Metrics
# BLEU + chrF + COMET
# ==========================================================

print("\nStep 3: Generating TEST translations...")

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


# ==========================================================
# Print Translation Metrics
# ==========================================================

print("\n===================================")
print("TEST Translation Metrics")
print("===================================")

print(f"BLEU Score  : {bleu:.6f}")
print(f"chrF Score  : {chrf:.6f}")
print(f"COMET Score : {comet:.6f}")

print("===================================")


# ==========================================================
# Save Evaluation JSON
# ==========================================================

print("\nStep 4: Saving evaluation.json...")


save_evaluation_json(

    bleu,

    chrf,

    comet,

    loss,

    accuracy,

    history,

    "models"

)


print("evaluation.json saved.")


# ==========================================================
# Finished
# ==========================================================

print("\n===================================")
print("Training Completed Successfully!")
print("===================================")


print("\nGenerated Files")

print("✔ encoder.keras")
print("✔ decoder.keras")
print("✔ seq2seq.keras")
print("✔ training_loss.png")
print("✔ training_accuracy.png")
print("✔ translation_metrics.txt")
print("✔ bleu_score.txt")
print("✔ evaluation.json")
print("✔ model_summary.txt")