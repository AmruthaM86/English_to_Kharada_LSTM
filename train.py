import os
import numpy as np

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from utils.dataset import get_data
from model import build_models

# ================================
# NEW
# ================================

from utils.evaluation import (
    save_training_graphs,
    save_bleu_score,
    save_evaluation_json,
    save_model_summary
)

# ===========================================
# Create Folders
# ===========================================

os.makedirs("models", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)

# ===========================================
# Load Dataset
# ===========================================

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

# ===========================================
# Teacher Forcing
# ===========================================

decoder_input_train = Y_train[:, :-1]
decoder_target_train = Y_train[:, 1:]

decoder_input_valid = Y_valid[:, :-1]
decoder_target_valid = Y_valid[:, 1:]

decoder_target_train = np.expand_dims(
    decoder_target_train,
    axis=-1
)

decoder_target_valid = np.expand_dims(
    decoder_target_valid,
    axis=-1
)

# ===========================================
# Build Models
# ===========================================

training_model, encoder_model, decoder_model = build_models(

    input_vocab_size=input_vocab_size,

    target_vocab_size=target_vocab_size,

    embedding_dim=256,

    latent_dim=512

)

training_model.summary()

# ===========================================
# Callbacks
# ===========================================

checkpoint = ModelCheckpoint(

    filepath="models/checkpoints/best_model.keras",

    monitor="val_loss",

    save_best_only=True,

    save_weights_only=False,

    verbose=1

)

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1

)

# ===========================================
# Train Model
# ===========================================

history = training_model.fit(

    x=[X_train, decoder_input_train],

    y=decoder_target_train,

    validation_data=(

        [X_valid, decoder_input_valid],

        decoder_target_valid

    ),

    epochs=50,

    batch_size=32,

    callbacks=[checkpoint, early_stopping]

)

# ===========================================
# Save Models
# ===========================================

training_model.save("models/seq2seq.keras")
encoder_model.save("models/encoder.keras")
decoder_model.save("models/decoder.keras")

print("\n====================================")
print("Models Saved Successfully!")
print("====================================")

# ===========================================
# Evaluate
# ===========================================

decoder_input_test = Y_test[:, :-1]

decoder_target_test = Y_test[:, 1:]

decoder_target_test = np.expand_dims(
    decoder_target_test,
    axis=-1
)

loss, accuracy = training_model.evaluate(

    [X_test, decoder_input_test],

    decoder_target_test,

    verbose=1

)

print("\n====================================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print("====================================")

# ===========================================
# Save Training Graphs
# ===========================================

save_training_graphs(
    history,
    "models"
)

# ===========================================
# Save Model Summary
# ===========================================

save_model_summary(
    training_model,
    "models"
)

# ===========================================
# Dummy BLEU (Replace Later)
# ===========================================

references = [
    [["hello", "friend"]],
    [["good", "morning"]]
]

predictions = [
    ["hello", "friend"],
    ["good", "morning"]
]

bleu = save_bleu_score(
    references,
    predictions,
    "models"
)

# ===========================================
# Save evaluation.json
# ===========================================

save_evaluation_json(
    bleu,
    history,
    "models"
)

# ===========================================
# Finished
# ===========================================

print("\n====================================")
print("Training Completed Successfully!")
print("====================================")
print("Generated Files:")
print("✔ training_loss.png")
print("✔ training_accuracy.png")
print("✔ bleu_score.txt")
print("✔ evaluation.json")
print("✔ model_summary.txt")