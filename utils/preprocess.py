import os
import random
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

SOURCE_FILE = os.path.join(
    DATASET_DIR,
    "english_kharada.txt"
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


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_english(text):
    """
    Clean English sentence.
    """

    text = text.lower()

    # Remove extra spaces
    text = " ".join(text.split())

    return text.strip()


def clean_kharada(text):
    """
    Clean Kharada/Kannada sentence.
    """

    # Remove extra spaces
    text = " ".join(text.split())

    return text.strip()


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(filepath):

    data = []

    print("Reading dataset:")
    print(filepath)

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # ------------------------------------------------
            # Expected format:
            #
            # Kharada sentence <start> English sentence <end>
            # ------------------------------------------------

            if "<start>" not in line:
                continue

            if "<end>" not in line:
                continue

            try:

                # Split source and target
                kharada_text, english_text = line.split(
                    "<start>",
                    1
                )

                # Remove everything after <end>
                english_text = english_text.split(
                    "<end>",
                    1
                )[0]

                # Clean both sentences
                kharada_text = clean_kharada(
                    kharada_text
                )

                english_text = clean_english(
                    english_text
                )

                # Make sure both sides contain text
                if kharada_text and english_text:

                    data.append(
                        (
                            kharada_text,
                            english_text
                        )
                    )

            except Exception as e:

                print(
                    f"Skipping invalid line "
                    f"{line_number}: {e}"
                )

    print(
        f"Total valid sentence pairs: {len(data)}"
    )

    return data


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(data, filepath):

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        for source, target in data:

            # Save as:
            #
            # Kharada<TAB>English
            #

            file.write(
                source
                + "\t"
                + target
                + "\n"
            )


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset():

    print("=" * 60)
    print("Kharada → English Dataset Preprocessing")
    print("=" * 60)

    # Create dataset folder if required
    os.makedirs(
        DATASET_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load complete dataset
    # --------------------------------------------------------

    data = load_dataset(
        SOURCE_FILE
    )

    total_samples = len(data)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if total_samples == 0:

        print()
        print("ERROR: No valid sentence pairs found.")
        print()
        print("Expected format:")
        print(
            "Kharada sentence "
            "<start> "
            "English sentence "
            "<end>"
        )

        return

    print()
    print(
        f"Total sentences available: "
        f"{total_samples}"
    )

    # --------------------------------------------------------
    # Shuffle dataset
    # --------------------------------------------------------

    random.seed(42)

    random.shuffle(data)

    # ========================================================
    # 80% TRAIN
    # 20% TEMPORARY
    # ========================================================

    train_data, temp_data = train_test_split(
        data,
        test_size=0.20,
        random_state=42
    )

    # ========================================================
    # TEMPORARY 20%
    #
    # Split it equally:
    #
    # 50% of 20% = 10% validation
    # 50% of 20% = 10% testing
    # ========================================================

    validation_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=42
    )

    # ========================================================
    # SAVE DATASETS
    # ========================================================

    save_dataset(
        train_data,
        TRAIN_FILE
    )

    save_dataset(
        validation_data,
        VALIDATION_FILE
    )

    save_dataset(
        test_data,
        TEST_FILE
    )

    # ========================================================
    # CALCULATE ACTUAL PERCENTAGES
    # ========================================================

    train_percentage = (
        len(train_data)
        / total_samples
        * 100
    )

    validation_percentage = (
        len(validation_data)
        / total_samples
        * 100
    )

    test_percentage = (
        len(test_data)
        / total_samples
        * 100
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("DATASET SPLIT COMPLETED")
    print("=" * 60)

    print(
        f"Total samples      : "
        f"{total_samples}"
    )

    print(
        f"Training samples   : "
        f"{len(train_data)} "
        f"({train_percentage:.2f}%)"
    )

    print(
        f"Validation samples : "
        f"{len(validation_data)} "
        f"({validation_percentage:.2f}%)"
    )

    print(
        f"Testing samples    : "
        f"{len(test_data)} "
        f"({test_percentage:.2f}%)"
    )

    print()
    print(
        f"Total after split  : "
        f"{len(train_data) + len(validation_data) + len(test_data)}"
    )

    print()
    print("Files created:")

    print(
        f"✔ {TRAIN_FILE}"
    )

    print(
        f"✔ {VALIDATION_FILE}"
    )

    print(
        f"✔ {TEST_FILE}"
    )

    print()
    print("=" * 60)
    print("Preprocessing Completed Successfully")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    split_dataset()