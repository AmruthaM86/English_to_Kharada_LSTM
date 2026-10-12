import os
import json
import numpy as np
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sacrebleu.metrics import CHRF

from comet import download_model, load_from_checkpoint


# ============================================================
# SAVE TRAINING GRAPHS
# ============================================================

def save_training_graphs(history, model_dir):

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Loss Graph
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    loss_path = os.path.join(
        model_dir,
        "training_loss.png"
    )

    plt.savefig(
        loss_path
    )

    plt.close()

    # --------------------------------------------------------
    # Accuracy Graph
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    accuracy_path = os.path.join(
        model_dir,
        "training_accuracy.png"
    )

    plt.savefig(
        accuracy_path
    )

    plt.close()


# ============================================================
# SAVE MODEL SUMMARY
# ============================================================

def save_model_summary(
    model,
    model_dir
):

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    summary_path = os.path.join(
        model_dir,
        "model_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        model.summary(
            print_fn=lambda x:
            f.write(x + "\n")
        )


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

def save_evaluation_json(
    bleu,
    chrf,
    comet,
    test_loss,
    test_accuracy,
    history,
    model_dir
):

    evaluation = {

        # ----------------------------------------------------
        # TEST DATASET TRANSLATION METRICS
        # ----------------------------------------------------

        "BLEU Score": float(
            bleu
        ),

        "chrF Score": float(
            chrf
        ),

        "COMET Score": float(
            comet
        ),

        # ----------------------------------------------------
        # TEST DATASET MODEL METRICS
        # ----------------------------------------------------

        "Test Loss": float(
            test_loss
        ),

        "Test Accuracy": float(
            test_accuracy
        ),

        # ----------------------------------------------------
        # FINAL TRAINING METRICS
        # ----------------------------------------------------

        "Final Training Accuracy": float(
            history.history["accuracy"][-1]
        ),

        "Final Training Loss": float(
            history.history["loss"][-1]
        ),

        # ----------------------------------------------------
        # FINAL VALIDATION METRICS
        # ----------------------------------------------------

        "Final Validation Accuracy": float(
            history.history["val_accuracy"][-1]
        ),

        "Final Validation Loss": float(
            history.history["val_loss"][-1]
        )
    }

    evaluation_path = os.path.join(
        model_dir,
        "evaluation.json"
    )

    with open(
        evaluation_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=4
        )

    return evaluation


# ============================================================
# CONVERT TOKEN IDs TO TEXT
# ============================================================

def sequence_to_text(
    sequence,
    tokenizer
):

    words = []

    reverse_word_index = {
        index: word
        for word, index
        in tokenizer.word_index.items()
    }

    for token_id in sequence:

        token_id = int(
            token_id
        )

        # Ignore padding
        if token_id == 0:
            continue

        word = reverse_word_index.get(
            token_id,
            ""
        )

        # Ignore special tokens
        if word in [
            "<start>",
            "<end>"
        ]:
            continue

        if word:

            words.append(
                word
            )

    return " ".join(
        words
    )


# ============================================================
# TRANSLATE ONE SENTENCE
# ============================================================

def translate_sentence(
    input_sequence,
    encoder_model,
    decoder_model,
    input_tokenizer,
    target_tokenizer,
    max_decoder_length
):

    # --------------------------------------------------------
    # Encode input sentence
    # --------------------------------------------------------

    states = encoder_model.predict(
        input_sequence,
        verbose=0
    )

    # --------------------------------------------------------
    # Get special token IDs
    # --------------------------------------------------------

    start_token = target_tokenizer.word_index.get("<start>")
    end_token = target_tokenizer.word_index.get("<end>")

    if start_token is None:
        print("ERROR: <start> token not found.")
        return ""

    # --------------------------------------------------------
    # Start decoder with <start>
    # --------------------------------------------------------

    target_seq = np.array([[start_token]])

    translated_words = []

    # --------------------------------------------------------
    # Generate words one by one
    # --------------------------------------------------------

    for step in range(max_decoder_length):

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states,
            verbose=0
        )

        sampled_token_index = int(
            np.argmax(output_tokens[0, -1, :])
        )

        # ----------------------------------------------------
        # DEBUG: show first prediction
        # ----------------------------------------------------

        if step == 0:
            predicted_word = None

            for word, index in target_tokenizer.word_index.items():
                if index == sampled_token_index:
                    predicted_word = word
                    break

            print(
                f"DEBUG: first predicted token = "
                f"{sampled_token_index}, word = {predicted_word}"
            )

        # ----------------------------------------------------
        # Stop at <end>
        # ----------------------------------------------------

        if (
            end_token is not None
            and sampled_token_index == end_token
        ):
            break

        # ----------------------------------------------------
        # Ignore padding token
        # ----------------------------------------------------

        if sampled_token_index == 0:
            break

        # ----------------------------------------------------
        # Convert token ID to word
        # ----------------------------------------------------

        sampled_word = ""

        for word, index in target_tokenizer.word_index.items():

            if index == sampled_token_index:
                sampled_word = word
                break

        # ----------------------------------------------------
        # Add predicted word
        # ----------------------------------------------------

        if sampled_word:

            if sampled_word not in [
                "<start>",
                "<end>"
            ]:
                translated_words.append(sampled_word)

        # ----------------------------------------------------
        # Next decoder input
        # ----------------------------------------------------

        target_seq = np.array(
            [[sampled_token_index]]
        )

        # ----------------------------------------------------
        # Update LSTM states
        # ----------------------------------------------------

        states = [h, c]

    return " ".join(translated_words)

# ============================================================
# GENERATE TEST SET PREDICTIONS
# ============================================================

def generate_test_predictions(
    X_test,
    Y_test,
    encoder_model,
    decoder_model,
    input_tokenizer,
    target_tokenizer,
    max_input_length,
    max_decoder_length
):

    sources = []

    references = []

    predictions = []

    print()

    print(
        "Generating translations for TEST dataset..."
    )

    print(
        f"Total test sentences: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Process every test sentence
    # --------------------------------------------------------

    for i in range(
        len(X_test)
    ):

        # ----------------------------------------------------
        # Source Kharada sentence
        # ----------------------------------------------------

        source_text = sequence_to_text(
            X_test[i],
            input_tokenizer
        )

        # ----------------------------------------------------
        # Actual English reference
        # ----------------------------------------------------

        reference_text = sequence_to_text(
            Y_test[i],
            target_tokenizer
        )

        # ----------------------------------------------------
        # Prepare encoder input
        # ----------------------------------------------------

        input_sequence = pad_sequences(
            [X_test[i]],
            maxlen=max_input_length,
            padding="post"
        )

        # ----------------------------------------------------
        # Generate translation
        # ----------------------------------------------------

        predicted_text = translate_sentence(
            input_sequence,
            encoder_model,
            decoder_model,
            input_tokenizer,
            target_tokenizer,
            max_decoder_length
        )

        sources.append(
            source_text
        )

        references.append(
            reference_text
        )

        predictions.append(
            predicted_text
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (i + 1) % 10 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(X_test)} "
                f"test sentences"
            )

    print(
        "Test translations generated successfully."
    )

    # ========================================================
    # TRANSLATION DIAGNOSTICS
    # ========================================================

    print()
    print("=" * 70)
    print("TRANSLATION DIAGNOSTICS")
    print("=" * 70)

    empty_predictions = sum(
        1
        for prediction in predictions
        if not prediction.strip()
    )

    print(
        f"Empty predictions : "
        f"{empty_predictions}/{len(predictions)}"
    )

    print(
        f"Non-empty predictions : "
        f"{len(predictions) - empty_predictions}/"
        f"{len(predictions)}"
    )

    # ========================================================
    # SAMPLE TRANSLATIONS
    # ========================================================

    print()
    print("=" * 70)
    print("SAMPLE TEST TRANSLATIONS")
    print("=" * 70)

    for i in range(
        min(10, len(predictions))
    ):

        print()
        print(
            f"Test Sentence {i + 1}"
        )

        print(
            "-" * 50
        )

        print(
            "Input     :",
            sources[i]
        )

        print(
            "Expected  :",
            references[i]
        )

        print(
            "Predicted :",
            predictions[i]
        )

    # ========================================================
    # SAVE ALL TEST PREDICTIONS
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    prediction_file = os.path.join(
        "models",
        "test_predictions.txt"
    )

    with open(
        prediction_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TEST DATASET TRANSLATION RESULTS\n"
        )

        f.write(
            "=================================\n\n"
        )

        for i in range(
            len(predictions)
        ):

            f.write(
                f"Test Sentence {i + 1}\n"
            )

            f.write(
                f"Input     : {sources[i]}\n"
            )

            f.write(
                f"Expected  : {references[i]}\n"
            )

            f.write(
                f"Predicted : {predictions[i]}\n"
            )

            f.write(
                "\n"
            )

    print()

    print(
        "All test predictions saved to:"
    )

    print(
        prediction_file
    )

    return (
        sources,
        references,
        predictions
    )


# ============================================================
# BLEU SCORE
# ============================================================

def calculate_bleu(
    references,
    predictions
):

    reference_tokens = []

    prediction_tokens = []

    for reference, prediction in zip(
        references,
        predictions
    ):

        reference_tokens.append(
            [
                reference.split()
            ]
        )

        prediction_tokens.append(
            prediction.split()
        )

    # --------------------------------------------------------
    # Corpus BLEU
    # --------------------------------------------------------

    bleu = corpus_bleu(
        reference_tokens,
        prediction_tokens
    )

    return float(
        bleu
    )


# ============================================================
# chrF SCORE
# ============================================================

def calculate_chrf(
    references,
    predictions
):

    chrf_metric = CHRF()

    result = chrf_metric.corpus_score(
        predictions,
        [
            references
        ]
    )

    return float(
        result.score
    )


# ============================================================
# COMET SCORE
# ============================================================

def calculate_comet(
    sources,
    references,
    predictions
):

    print()

    print(
        "Loading COMET model..."
    )

    # --------------------------------------------------------
    # Download/load pretrained COMET model
    # --------------------------------------------------------

    model_path = download_model(
        "Unbabel/wmt22-comet-da"
    )

    comet_model = load_from_checkpoint(
        model_path
    )

    # --------------------------------------------------------
    # Prepare COMET input
    # --------------------------------------------------------

    data = []

    for source, prediction, reference in zip(
        sources,
        predictions,
        references
    ):

        data.append(
            {
                "src": source,
                "mt": prediction,
                "ref": reference
            }
        )

    print(
        "Calculating COMET score..."
    )

    # --------------------------------------------------------
    # Calculate COMET
    # --------------------------------------------------------

    output = comet_model.predict(
        data,
        batch_size=8,
        gpus=0
    )

    comet_score = output.system_score

    return float(
        comet_score
    )


# ============================================================
# GENERATE ALL TRANSLATION METRICS
# ============================================================

def generate_translation_metrics(
    X_test,
    Y_test,
    encoder_model,
    decoder_model,
    input_tokenizer,
    target_tokenizer,
    max_input_length,
    max_decoder_length,
    model_dir
):

    # ========================================================
    # STEP 1
    # Generate translations ONLY from TEST data
    # ========================================================

    (
        sources,
        references,
        predictions
    ) = generate_test_predictions(

        X_test,

        Y_test,

        encoder_model,

        decoder_model,

        input_tokenizer,

        target_tokenizer,

        max_input_length,

        max_decoder_length

    )

    # ========================================================
    # STEP 2
    # BLEU
    # ========================================================

    print()

    print(
        "Calculating BLEU score..."
    )

    bleu = calculate_bleu(
        references,
        predictions
    )

    print(
        f"BLEU Score: {bleu:.6f}"
    )

    # ========================================================
    # STEP 3
    # chrF
    # ========================================================

    print()

    print(
        "Calculating chrF score..."
    )

    chrf = calculate_chrf(
        references,
        predictions
    )

    print(
        f"chrF Score: {chrf:.6f}"
    )

    # ========================================================
    # STEP 4
    # COMET
    # ========================================================

    print()

    comet = calculate_comet(
        sources,
        references,
        predictions
    )

    print(
        f"COMET Score: {comet:.6f}"
    )

    # ========================================================
    # STEP 5
    # Save translation metrics
    # ========================================================

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    metrics_path = os.path.join(
        model_dir,
        "translation_metrics.txt"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TEST DATASET TRANSLATION METRICS\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            f"Number of Test Sentences : "
            f"{len(predictions)}\n\n"
        )

        f.write(
            f"BLEU Score  : "
            f"{bleu:.6f}\n"
        )

        f.write(
            f"chrF Score  : "
            f"{chrf:.6f}\n"
        )

        f.write(
            f"COMET Score : "
            f"{comet:.6f}\n"
        )

    print()

    print(
        "Translation metrics saved to:"
    )

    print(
        metrics_path
    )

    return (
        bleu,
        chrf,
        comet
    )