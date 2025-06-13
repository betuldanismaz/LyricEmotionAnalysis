# LyricEmotionAnalysis

## Description

This project uses AI models to predict the primary emotion (happy, sad, nostalgic, energetic) expressed in a song based on its lyrics. It utilizes Arousal/Valence (A/V) values and quadrant information to improve emotion mapping and enhance the performance of various classification models.

## Key Features

*   **Emotion Mapping:**  A sophisticated rule-based system that assigns emotions based on Arousal/Valence values, quadrant information, and (optionally) mood keywords.
*   **Multiple Models:** Implements and compares the performance of TF-IDF with Logistic Regression, TF-IDF with SVM, and fine-tuning a pre-trained BERT model.
*   **Data Preprocessing:** Includes advanced text preprocessing techniques for English lyrics, including stop word removal and lemmatization.
*   **Detailed Analysis:** Provides comprehensive evaluation metrics, including accuracy, precision, recall, F1-score, and confusion matrices for each model.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/betuldanismaz/LyricEmotionAnalysis.git
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

3.  **Obtain the data:**  The project requires the following data files:

    *   `merge_lyrics_balanced_metadata.csv`
    *   `merge_lyrics_balanced_av_values.csv`
    *   Lyrics files organized in `Q1`, `Q2`, `Q3`, and `Q4` folders.


4.  **Configure the project:**

    *   Open `finalproje_colab_grup25.py` in Google Colab.
    *   In **Cell 2 (Configuration and File Paths)**, update the `DRIVE_PROJECT_BASE` variable to point to the location of your data files in Google Drive.  Adjust other configuration parameters as needed (TVT ratio, BERT settings, etc.).

## Usage

1.  **Open the `lyric-emotion-analysis.py` notebook in Google Colab.**

2.  **Connect to Google Drive:**  Ensure the notebook is connected to your Google Drive account.

3.  **Run the notebook cells sequentially.**  The notebook is designed to be executed from top to bottom.  Pay attention to the output and warnings in each cell.

4.  **Analyze Results:**  After the notebook completes, the results (performance metrics, confusion matrices, and graphs) will be displayed.

## Data Files

The project uses the following data files:

*   **`merge_lyrics_balanced_metadata.csv`:**  Contains metadata for the songs, including song ID, quadrant, and (optionally) moods.
*   **`merge_lyrics_balanced_av_values.csv`:**  Contains Arousal and Valence values for each song.
*   **`Q1`, `Q2`, `Q3`, `Q4` folders:** Contain the lyrics files for each song, organized by quadrant.

## Models

The project implements the following machine learning models:

*   **TF-IDF + Logistic Regression:**
    *   **Features:**  TF-IDF (Term Frequency-Inverse Document Frequency) vectors of the lyrics.
    *   **Classifier:** Logistic Regression with `liblinear` solver, C=1.0, `multi_class='ovr'`, `random_state=42`, and `class_weight='balanced'`.

*   **TF-IDF + SVM:**
    *   **Features:** TF-IDF vectors of the lyrics.
    *   **Classifier:**  Support Vector Machine with a linear kernel, C=0.1, `random_state=42`, `class_weight='balanced'`, and `probability=False`.

*   **BERT (bert-base-multilingual-cased):**
    *   Fine-tuned pre-trained BERT model for sequence classification.
    *   **Hyperparameters:** `BERT_MAX_LEN=128`, `BERT_BATCH_SIZE=8`, `BERT_EPOCHS=3`

## Emotion Mapping Strategy

The emotion mapping is performed in **Cell 4** of the notebook. It's a key part of the project and combines:

*   **Arousal/Valence (A/V) Values:**  Songs are positioned on a 2D plane based on these values.
*   **Quadrants:** The A/V plane is divided into four quadrants, each associated with general emotional tendencies.
*   **Mood Keywords (Optional):** If available, mood keywords from the metadata provide additional clues about the song's emotion.
The `map_emotion_v9` function assigns emotions based on these factors, using carefully tuned thresholds and priority rules.

##  Google Colab Specifics

*   To utilize GPU acceleration for BERT training, set "Runtime type" in Colab to "GPU" (Runtime -> Change runtime type -> Hardware accelerator).

*   Update `DRIVE_PROJECT_BASE` (Cell 2) to match your data directory.
