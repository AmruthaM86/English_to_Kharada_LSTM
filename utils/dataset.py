
import os
import json
import joblib
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

TRAIN_FILE = os.path.join(
    DATASET_DIR,
    "train.txt"
)

VALIDATION_FILE = os.path.join(
    DATASET_DIR,
    "validation.txt"
)

TEST_FILE = os.path.join(
    DATASET_DIR,
    "test.txt"
)

TOKENIZER_DIR = os.path.join(
    BASE_DIR,
    "tokenizer"
)


# ============================================================
# LOAD FILE
# ============================================================

def load_file(filepath):

    sources = []
    targets = []

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                continue

            source = parts[0].strip()
            target = parts[1].strip()

            if source and target:

                sources.append(source)
                targets.append(target)

    return sources, targets


# ============================================================
# GET DATA
# ============================================================

def get_data():

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)


    # --------------------------------------------------------
    # Load train data
    # --------------------------------------------------------

    train_sources, train_targets = load_file(
        TRAIN_FILE
    )


    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    valid_sources, valid_targets = load_file(
        VALIDATION_FILE
    )


    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_sources, test_targets = load_file(
        TEST_FILE
    )


    print(
        f"Training pairs   : {len(train_sources)}"
    )

    print(
        f"Validation pairs : {len(valid_sources)}"
    )

    print(
        f"Test pairs       : {len(test_sources)}"
    )


    # ========================================================
    # ADD SPECIAL TOKENS TO TARGET SENTENCES
    # ========================================================

    train_targets = [
        "<start> " + text + " <end>"
        for text in train_targets
    ]

    valid_targets = [
        "<start> " + text + " <end>"
        for text in valid_targets
    ]

    test_targets = [
        "<start> " + text + " <end>"
        for text in test_targets
    ]


    # ========================================================
    # SOURCE TOKENIZER
    # ========================================================

    input_tokenizer = Tokenizer(
        filters="",
        lower=False,
        oov_token="<unk>"
    )

    input_tokenizer.fit_on_texts(
        train_sources
    )


    # ========================================================
    # TARGET TOKENIZER
    # ========================================================

    target_tokenizer = Tokenizer(
        filters="",
        lower=True,
        oov_token="<unk>"
    )

    target_tokenizer.fit_on_texts(
        train_targets
    )


    # ========================================================
    # CONVERT TEXT TO SEQUENCES
    # ========================================================

    X_train = input_tokenizer.texts_to_sequences(
        train_sources
    )

    X_valid = input_tokenizer.texts_to_sequences(
        valid_sources
    )

    X_test = input_tokenizer.texts_to_sequences(
        test_sources
    )


    Y_train = target_tokenizer.texts_to_sequences(
        train_targets
    )

    Y_valid = target_tokenizer.texts_to_sequences(
        valid_targets
    )

    Y_test = target_tokenizer.texts_to_sequences(
        test_targets
    )


    # ========================================================
    # FIND MAXIMUM LENGTHS
    # ========================================================

    max_encoder_length = max(
        len(sequence)
        for sequence in X_train
    )

    max_decoder_length = max(
        len(sequence)
        for sequence in Y_train
    )


    # ========================================================
    # PAD SOURCE SEQUENCES
    # ========================================================

    X_train = pad_sequences(
        X_train,
        maxlen=max_encoder_length,
        padding="post"
    )

    X_valid = pad_sequences(
        X_valid,
        maxlen=max_encoder_length,
        padding="post"
    )

    X_test = pad_sequences(
        X_test,
        maxlen=max_encoder_length,
        padding="post"
    )


    # ========================================================
    # PAD TARGET SEQUENCES
    # ========================================================

    Y_train = pad_sequences(
        Y_train,
        maxlen=max_decoder_length,
        padding="post"
    )

    Y_valid = pad_sequences(
        Y_valid,
        maxlen=max_decoder_length,
        padding="post"
    )

    Y_test = pad_sequences(
        Y_test,
        maxlen=max_decoder_length,
        padding="post"
    )


    # ========================================================
    # VOCABULARY SIZES
    # ========================================================

    input_vocab_size = (
        len(input_tokenizer.word_index) + 1
    )

    target_vocab_size = (
        len(target_tokenizer.word_index) + 1
    )


    # ========================================================
    # SAVE TOKENIZERS AND LENGTHS
    # ========================================================

    os.makedirs(
        TOKENIZER_DIR,
        exist_ok=True
    )

    joblib.dump(
        input_tokenizer,
        os.path.join(
            TOKENIZER_DIR,
            "input_tokenizer.pkl"
        )
    )

    joblib.dump(
        target_tokenizer,
        os.path.join(
            TOKENIZER_DIR,
            "target_tokenizer.pkl"
        )
    )

    with open(
        os.path.join(
            TOKENIZER_DIR,
            "lengths.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "max_encoder_length": max_encoder_length,
                "max_decoder_length": max_decoder_length
            },
            file,
            indent=4
        )

    print()
    print("Tokenizers and lengths saved successfully.")


    # ========================================================
    # DISPLAY SHAPES
    # ========================================================

    print()
    print("=" * 60)
    print("Final Array Shapes")
    print("=" * 60)

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "Y_train:",
        Y_train.shape
    )

    print(
        "X_valid:",
        X_valid.shape
    )

    print(
        "Y_valid:",
        Y_valid.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    print(
        "Y_test :",
        Y_test.shape
    )


    # ========================================================
    # DATA CARDINALITY CHECK
    # ========================================================

    assert len(X_train) == len(Y_train)

    assert len(X_valid) == len(Y_valid)

    assert len(X_test) == len(Y_test)


    print()
    print(
        "Dataset cardinality check: PASSED"
    )


    # ========================================================
    # SPECIAL TOKEN CHECK
    # ========================================================

    start_token = target_tokenizer.word_index.get(
        "<start>"
    )

    end_token = target_tokenizer.word_index.get(
        "<end>"
    )


    print()
    print(
        "START TOKEN:",
        start_token
    )

    print(
        "END TOKEN:",
        end_token
    )


    # ========================================================
    # RETURN DATA
    # ========================================================

    return (
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
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    get_data()