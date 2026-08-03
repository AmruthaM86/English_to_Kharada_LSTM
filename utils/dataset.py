import os
import joblib
import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train.txt")
VALID_PATH = os.path.join(BASE_DIR, "dataset", "validation.txt")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "test.txt")

TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

os.makedirs(TOKENIZER_DIR, exist_ok=True)

# ==========================================================
# Read Dataset
# ==========================================================

def load_file(path):

    kharada = []
    english = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                continue

            # Dataset format:
            # Kharada<TAB><start> English <end>

            kharada.append(parts[0])
            english.append(parts[1])

    # IMPORTANT
    return kharada, english


# ==========================================================
# Load Data
# ==========================================================

train_kha, train_eng = load_file(TRAIN_PATH)
valid_kha, valid_eng = load_file(VALID_PATH)
test_kha, test_eng = load_file(TEST_PATH)

# ==========================================================
# Tokenizers
# ==========================================================

input_tokenizer = Tokenizer(
    filters=""
)

input_tokenizer.fit_on_texts(train_kha)

target_tokenizer = Tokenizer(
    filters=""
)

target_tokenizer.fit_on_texts(train_eng)

# ==========================================================
# Save Tokenizers
# ==========================================================

joblib.dump(
    input_tokenizer,
    os.path.join(TOKENIZER_DIR, "input_tokenizer.pkl")
)

joblib.dump(
    target_tokenizer,
    os.path.join(TOKENIZER_DIR, "target_tokenizer.pkl")
)

# ==========================================================
# Convert to Sequences
# ==========================================================

train_encoder = input_tokenizer.texts_to_sequences(train_kha)
valid_encoder = input_tokenizer.texts_to_sequences(valid_kha)
test_encoder = input_tokenizer.texts_to_sequences(test_kha)

train_decoder = target_tokenizer.texts_to_sequences(train_eng)
valid_decoder = target_tokenizer.texts_to_sequences(valid_eng)
test_decoder = target_tokenizer.texts_to_sequences(test_eng)

# ==========================================================
# Maximum Length
# ==========================================================

max_encoder_length = max(len(x) for x in train_encoder)

max_decoder_length = max(len(x) for x in train_decoder)

length_info = {

    "max_encoder_length": max_encoder_length,

    "max_decoder_length": max_decoder_length

}

with open(

    os.path.join(TOKENIZER_DIR, "lengths.json"),

    "w"

) as f:

    json.dump(length_info, f)

# ==========================================================
# Padding
# ==========================================================

X_train = pad_sequences(
    train_encoder,
    maxlen=max_encoder_length,
    padding="post"
)

X_valid = pad_sequences(
    valid_encoder,
    maxlen=max_encoder_length,
    padding="post"
)

X_test = pad_sequences(
    test_encoder,
    maxlen=max_encoder_length,
    padding="post"
)

Y_train = pad_sequences(
    train_decoder,
    maxlen=max_decoder_length,
    padding="post"
)

Y_valid = pad_sequences(
    valid_decoder,
    maxlen=max_decoder_length,
    padding="post"
)

Y_test = pad_sequences(
    test_decoder,
    maxlen=max_decoder_length,
    padding="post"
)

# ==========================================================
# Vocabulary Sizes
# ==========================================================

input_vocab_size = len(input_tokenizer.word_index) + 1

target_vocab_size = len(target_tokenizer.word_index) + 1

# ==========================================================
# Reverse Dictionary
# ==========================================================

reverse_target_index = {

    value: key

    for key, value in target_tokenizer.word_index.items()

}

# ==========================================================
# Return Data
# ==========================================================

def get_data():

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