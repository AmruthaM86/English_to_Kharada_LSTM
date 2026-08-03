import os
import re
import random
from sklearn.model_selection import train_test_split

# -------------------------------
# File Paths
# -------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "english_kharada.txt")

TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train.txt")
VALID_PATH = os.path.join(BASE_DIR, "dataset", "validation.txt")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "test.txt")


# -------------------------------
# Text Cleaning
# -------------------------------

def clean_english(sentence):

    sentence = sentence.lower().strip()
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence


def clean_kharada(sentence):

    sentence = sentence.strip()
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence


# -------------------------------
# Read Dataset
# -------------------------------

def load_dataset():

    pairs = []

    with open(DATASET_PATH, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                continue

            # Dataset format:
            # Kharada<TAB>English

            kharada = clean_kharada(parts[0])

            english = clean_english(parts[1])

            # Decoder target (English)
            english = "<start> " + english + " <end>"

            pairs.append((kharada, english))

    return pairs


# -------------------------------
# Save Dataset
# -------------------------------

def save_dataset(data, filepath):

    with open(filepath, "w", encoding="utf-8") as file:

        for kharada, english in data:

            file.write(kharada + "\t" + english + "\n")


# -------------------------------
# Split Dataset
# -------------------------------

def split_dataset():

    data = load_dataset()

    random.shuffle(data)

    train_data, temp_data = train_test_split(
        data,
        test_size=0.20,
        random_state=42
    )

    valid_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=42
    )

    save_dataset(train_data, TRAIN_PATH)
    save_dataset(valid_data, VALID_PATH)
    save_dataset(test_data, TEST_PATH)

    print("=" * 50)
    print("Dataset Successfully Split")
    print("=" * 50)
    print("Total :", len(data))
    print("Train :", len(train_data))
    print("Validation :", len(valid_data))
    print("Test :", len(test_data))


if __name__ == "__main__":
    split_dataset()