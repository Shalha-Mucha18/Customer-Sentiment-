from flask import Flask, render_template, request
import mlflow
import pickle
import numpy as np
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub  
from pathlib import Path

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_DIR = ROOT_DIR / "models"


def lematization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan   

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lematization(text)
    return text
             
# MLflow tracking configuration
dagshub_token = os.getenv("CAPSTONE_TEST")
if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri('https://dagshub.com/Shalha-Mucha18/Capstone-Project-2026.mlflow')
    dagshub.init(repo_owner='Shalha-Mucha18', repo_name='Capstone-Project-2026', mlflow=True)
else:
    mlflow.set_tracking_uri(f"file://{ROOT_DIR / 'mlruns'}")



# Initializing the Flask application
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)


model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    try:
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
        if latest_version:
            return latest_version[0].version
    except Exception:
        # Registry may be unavailable (e.g., local file store). Fall back to search.
        pass
    # Fallback: pick the highest version across all stages
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return None
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version)).version

model_version = get_latest_model_version(model_name)
if model_version:
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Loading model from URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
else:
    print("No registered model version found. Falling back to local models/model.pkl")
    model = pickle.load(open(MODEL_DIR / "model.pkl", "rb"))

vectorizer = pickle.load(open(MODEL_DIR / "vectorizer.pkl", "rb"))



# Routes

@app.route("/")
def home():

    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None, error=None, text_value="")
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():

    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()    

    text = request.form.get("text", "")
    if not text or not text.strip():
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return render_template(
            "index.html",
            error="Please enter some text to analyze.",
            text_value="",
            result=None,
        )
    # Clean text
    cleaned_text = normalize_text(text)
    # Vectorize text
    feature = vectorizer.transform([cleaned_text])
    features_df = pd.DataFrame(feature.toarray(), columns=vectorizer.get_feature_names_out())
    # Predict

    result  = model.predict(features_df)

    prediction = result[0]

    # Increment prediction count metric
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    # Measure latency

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction, text_value=text)

@app.route("/metrics", methods=["GET"])
def metrics():
    return  generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(debug=True)
