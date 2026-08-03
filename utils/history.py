import os
import pandas as pd
from datetime import datetime

# ==========================================
# Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HISTORY_DIR = os.path.join(BASE_DIR, "history")

os.makedirs(HISTORY_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(
    HISTORY_DIR,
    "translation_history.csv"
)

# ==========================================
# Create History File
# ==========================================

def initialize_history():

    if not os.path.exists(HISTORY_FILE):

        df = pd.DataFrame(

            columns=[

                "Date",

                "Time",

                "Kharada",

                "English"

            ]

        )

        df.to_csv(

            HISTORY_FILE,

            index=False,

            encoding="utf-8"

        )

# ==========================================
# Save Translation
# ==========================================

def save_history(kharada, english):

    initialize_history()

    now = datetime.now()

    date = now.strftime("%d-%m-%Y")

    time = now.strftime("%H:%M:%S")

    row = pd.DataFrame({

        "Date":[date],

        "Time":[time],

        "Kharada":[kharada],

        "English":[english]

    })

    row.to_csv(

        HISTORY_FILE,

        mode="a",

        header=False,

        index=False,

        encoding="utf-8"

    )

# ==========================================
# Read History
# ==========================================

def load_history():

    initialize_history()

    return pd.read_csv(

        HISTORY_FILE,

        encoding="utf-8"

    )

# ==========================================
# Delete History
# ==========================================

def clear_history():

    df = pd.DataFrame(

        columns=[

            "Date",

            "Time",

            "Kharada",

            "English"

        ]

    )

    df.to_csv(

        HISTORY_FILE,

        index=False,

        encoding="utf-8"

    )

# ==========================================
# Search History
# ==========================================

def search_history(keyword):

    df = load_history()

    keyword = keyword.lower()

    result = df[

        df["Kharada"].str.lower().str.contains(keyword)

        |

        df["English"].str.lower().str.contains(keyword)

    ]

    return result

# ==========================================
# Total Translations
# ==========================================

def total_translations():

    df = load_history()

    return len(df)