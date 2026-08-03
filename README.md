<<<<<<< HEAD
# How to Run the Project

Follow the steps below to set up and run the English to Kharada Translation project.

---

## Step 1: Clone the Repository

```bash
git clone <repository_url>
cd English_to_Kharada_LSTM
```

Or simply download the project and extract it.

---

## Step 2: Install Python

Install **Python 3.10.11 (64-bit)** from:

https://www.python.org/downloads/release/python-31011/

Verify the installation:

```bash
py -3.10 --version
```

Expected Output:

```
Python 3.10.11
```

---

## Step 3: Create a Virtual Environment

Navigate to the project folder.

```bash
cd English_to_Kharada_LSTM
```

Create the virtual environment.

```bash
py -3.10 -m venv .venv
```

---

## Step 4: Activate the Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

You should see:

```
(.venv)
```

Verify the Python version.

```bash
python --version
```

Expected:

```
Python 3.10.11
```

---

## Step 5: Install Dependencies

Install all required packages.

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

Or install manually.

```bash
pip install tensorflow==2.17.1 numpy==1.26.4 pandas scikit-learn matplotlib streamlit nltk tqdm joblib
```

---

## Step 6: Prepare the Dataset

Place the dataset inside:

```
dataset/
```

Dataset format:

```
English Sentence<TAB>Kharada Sentence
```

Example:

```
Hello	ಹಲೋ
How are you?	ನೀವು ಹೇಗಿದ್ದೀರಿ?
Good Morning	ಶುಭೋದಯ
```

Split the dataset into Train, Validation and Test.

```bash
python utils/preprocess.py
```

Expected Output:

```
==================================================
Dataset Successfully Split
==================================================
Total : 952
Train : 761
Validation : 95
Test : 96
```

---

## Step 7: Train the Model

Run:

```bash
python train.py
```

The model will:

- Load the dataset
- Tokenize the text
- Create vocabularies
- Train the Seq2Seq LSTM model
- Save the best model
- Plot training graphs

Expected Output:

```
Epoch 1/50
...
Epoch 17/50

Models Saved Successfully!

Test Accuracy : 44%
```

The trained model will be saved inside:

```
models/
```

---

## Step 8: Test the Model

Run:

```bash
python inference.py
```

Example:

```
==================================================
English → Kharada Translator
==================================================

English :
I am happy today

Kharada :
ಹಾವ ಅಜ್ಜೇಂಟು ನಾ
```

Type **exit** to close the translator.

---

## Step 9: Run the Web Application

Launch Streamlit.

```bash
streamlit run app.py
```

Open the browser:

```
http://localhost:8501
```

Enter an English sentence and click **Translate**.

---

## Project Workflow

```
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Tokenization
   │
   ▼
Padding
   │
   ▼
Seq2Seq LSTM Model
   │
   ▼
Training
   │
   ▼
Saved Model
   │
   ▼
Inference
   │
   ▼
Streamlit Web Application
```

---

## Generated Files

After training, the following files are created automatically:

```
dataset/
    train.txt
    validation.txt
    test.txt

tokenizer/
    input_tokenizer.pkl
    target_tokenizer.pkl
    lengths.json

models/
    seq2seq.keras
    encoder.keras
    decoder.keras
    training_loss.png
    training_accuracy.png
```

---

## Requirements

- Python 3.10.11
- TensorFlow 2.17.1
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- NLTK

---

## Notes

- Always activate the virtual environment before running the project.
- Ensure the dataset uses **tab (`\t`)** as the separator between English and Kharada sentences.
- If a package is missing, install it using:

```bash
pip install <package_name>
```

Example:

```bash
pip install matplotlib
```




```
Kharada_English_Translator/
│
├── app.py                         # Main Streamlit App
├── train.py                       # Train the model
├── inference.py                   # Translation logic
├── model.py                       # Encoder Decoder LSTM model
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── english_kharada.txt
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
│
├── tokenizer/
│   ├── input_tokenizer.pkl
│   ├── target_tokenizer.pkl
│   └── lengths.json
│
├── models/
│   ├── seq2seq.keras
│   ├── encoder.keras
│   ├── decoder.keras
│   ├── checkpoints/
│   │      └── best_model.keras
│   │
│   ├── training_loss.png
│   ├── training_accuracy.png
│   ├── confusion_matrix.png
│   ├── bleu_score.txt
│   ├── evaluation.json
│   └── model_summary.txt
│
├── utils/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── helper.py
│   ├── evaluation.py
│   ├── metrics.py
│   └── history.py
│
├── pages/
│   ├── 1_Home.py
│   ├── 2_Translator.py
│   ├── 3_History.py
│   ├── 4_Model_Evaluation.py
│   ├── 5_Training_Graphs.py
│   ├── 6_Dataset_Statistics.py
│   ├── 7_About_Project.py
│   └── 8_Login.py
│
├── assets/
│   ├── logo.png
│   ├── banner.png
│   ├── background.jpg
│   ├── icons/
│   │      ├── home.png
│   │      ├── translate.png
│   │      ├── history.png
│   │      ├── graph.png
│   │      └── login.png
│   │
│   └── css/
│          └── style.css
│
├── history/
│   └── translation_history.csv
│
├── reports/
│   ├── project_report.pdf
│   ├── evaluation_report.pdf
│   └── screenshots/
│
└── .streamlit/
    └── config.toml
    ```

pip install nltk scikit-learn matplotlib pandas
=======
# English_to_Kharada_LSTM
LSTM model
>>>>>>> 03814c1bce3758addfacc2d1232f8b19cbcf9294
