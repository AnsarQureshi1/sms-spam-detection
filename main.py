from fastapi import FastAPI, HTTPException
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from schema import PredictionResponse, SMSRequest

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI()

ps = PorterStemmer()

def transform_text(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    vectorizer = None


@app.get("/")
def home():
    return {"message": "SMS Spam Detection API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_spam(request: SMSRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message Should not be empty")

    cleaned = transform_text(request.message)

    try:
        vector = vectorizer.transform([cleaned])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")

    prediction = model.predict(vector)[0]

    is_spam = bool(prediction == 1 or str(prediction).lower() in ["spam", "1"])

    return PredictionResponse(
        message=request.message,
        prediction="spam" if is_spam else "ham",
        is_spam=is_spam,
    )

