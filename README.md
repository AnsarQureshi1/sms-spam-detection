# SMS Spam Detection

A machine learning powered SMS spam classifier with a REST API built using **FastAPI**. The project trains a text classification model on the **SMS Spam Collection Dataset** and exposes it through a simple prediction endpoint.

---

## Overview

This project demonstrates an end-to-end NLP pipeline for detecting spam messages:

1. **Data acquisition** from the UCI SMS Spam Collection dataset.
2. **Exploratory data analysis** and cleaning.
3. **Text preprocessing** (lowercasing, tokenization, stopword removal, stemming).
4. **Feature extraction** with TF-IDF (3,000 features).
5. **Model training** with Naive Bayes classifiers.
6. **Deployment** as a FastAPI prediction service.

The trained artifacts (`model.pkl` and `vectorizer.pkl`) are bundled with the repository, so the API can be started without retraining.

---

## Features

- Multinomial Naive Bayes classifier trained on the SMS Spam Collection dataset.
- TF-IDF vectorization limited to the 3,000 most relevant features.
- Consistent text preprocessing between training and inference.
- FastAPI REST API with request/response validation via Pydantic.
- Interactive API documentation at `/docs` (Swagger UI) and `/redoc`.
- Clear error handling for empty input and missing model artifacts.

---

## Project Structure

```text
sms-spam-detection/
├── main.py                    # FastAPI application and preprocessing logic
├── schema.py                  # Pydantic request/response models
├── model.pkl                  # Trained Multinomial Naive Bayes model
├── vectorizer.pkl             # Fitted TF-IDF vectorizer
├── sms_spam_detection.ipynb   # EDA, training, and evaluation notebook
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## Dataset

The model is trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (UCI Machine Learning Repository), containing labeled SMS messages:

| Label | Description |
| ----- | ----------- |
| `ham` | Legitimate messages |
| `spam` | Unwanted/promotional messages |

During preprocessing, duplicate rows are removed and labels are encoded (`ham = 0`, `spam = 1`) before training.

---

## Tech Stack

| Component | Technology |
| --------- | ---------- |
| API framework | FastAPI |
| Data validation | Pydantic |
| ML / NLP | scikit-learn, NLTK |
| Text vectorization | TF-IDF |
| Model | Multinomial Naive Bayes |
| Serialization | joblib |
| Server | Uvicorn |

---

## Getting Started

### Prerequisites

- Python 3.10+ (developed on Python 3.14)
- pip

### Installation

```bash
git clone https://github.com/AnsarQureshi1/sms-spam-detection.git
cd sms-spam-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The required NLTK corpora (`punkt`, `punkt_tab`, `stopwords`) are downloaded automatically on the first run of `main.py`.

### Running the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once running:

- API root: <http://localhost:8000/>
- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

---

## API Reference

### `GET /`

Health check confirming the service is running.

**Response**

```json
{
  "message": "SMS Spam Detection API is running"
}
```

### `POST /predict`

Classifies a single SMS message as `spam` or `ham`.

**Request body**

```json
{
  "message": "Congratulations! You have won a free ticket. Call now to claim your prize."
}
```

**Response**

```json
{
  "message": "Congratulations! You have won a free ticket. Call now to claim your prize.",
  "prediction": "spam",
  "is_spam": true
}
```

**Example with cURL**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You have won a free ticket. Call now to claim your prize."}'
```

**Error responses**

| Status | Detail |
| ------ | ------ |
| `400` | Message should not be empty |
| `500` | Model not loaded / vectorization failure |

---

## Methodology

The notebook (`sms_spam_detection.ipynb`) covers the full workflow:

1. **Loading** the dataset and removing unused columns.
2. **Renaming** columns to `target` and `text`.
3. **Encoding** labels and dropping duplicate messages.
4. **EDA** including class distribution, message length statistics, correlation heatmap, and word clouds.
5. **Text transformation** using the shared `transform_text` pipeline:
   - Lowercasing
   - Tokenization
   - Alphanumeric filtering
   - Stopword and punctuation removal
   - Porter stemming
6. **Vectorization** with `TfidfVectorizer(max_features=3000)`.
7. **Model comparison** across Gaussian, Multinomial, and Bernoulli Naive Bayes.
8. **Saving** the best-performing vectorizer and model with `joblib`/`pickle`.

The deployed model is a **Multinomial Naive Bayes** classifier paired with a TF-IDF vectorizer.

---

## Model Artifacts

| File | Description |
| ---- | ----------- |
| `model.pkl` | Trained `MultinomialNB` classifier |
| `vectorizer.pkl` | Fitted `TfidfVectorizer` (3,000 features) |

> **Note:** The bundled artifacts were serialized with scikit-learn 1.6.1. When loading with a different scikit-learn version, a compatibility warning may appear. Retrain the model using the notebook to regenerate version-matched artifacts if needed.

---

## Retraining the Model

1. Open `sms_spam_detection.ipynb` in Jupyter Notebook or JupyterLab.
2. Install notebook dependencies (`pandas`, `matplotlib`, `seaborn`, `wordcloud`, `kagglehub`) if not already installed.
3. Run all cells sequentially. The dataset is downloaded automatically via `kagglehub`.
4. The notebook writes new `model.pkl` and `vectorizer.pkl` files into the project directory.
5. Restart the API server to load the updated artifacts.

---

## Limitations

- The model is trained on English SMS messages and may not generalize to other languages or domains.
- Naive Bayes assumes feature independence, which limits complex pattern learning.
- Predictions are based only on message text; sender metadata and URLs are not considered.
- The current API accepts one message per request. Batch prediction is not implemented.


