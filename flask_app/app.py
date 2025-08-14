import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import joblib
import re
import yaml
import traceback
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger.logging import logging
from src.exception.exception import customexception

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and vectorizer once at startup
def load_model_and_vectorizer():
    try:
        # Load parameters
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        logging.info("Successfully loaded params.yaml")

        # Configure MLflow tracking
        mlflow_params = params["mlflow"]
        mlflow.set_tracking_uri(
            f"https://{mlflow_params['username']}:{mlflow_params['token']}@dagshub.com/{mlflow_params['repo']}"
        )
        logging.info(f"Configured MLflow tracking URI for {mlflow_params['repo']}")

        # Load model from MLflow registry
        model_uri = f"models:/youtube_chromeplugin_model/2"
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("Successfully loaded model")

        # Load vectorizer from local file
        vectorizer_path = "artifacts/model_trainer/tfidf_vectorizer.pkl"
        vectorizer = joblib.load(vectorizer_path)
        logging.info("Successfully loaded vectorizer")

        return model, vectorizer

    except Exception as e:
        logging.error(f"Error loading model/vectorizer: {str(e)}")
        logging.error(traceback.format_exc())
        raise customexception(e, sys)

# Global load
model, vectorizer = load_model_and_vectorizer()

# Preprocessing
def preprocess_comment(comment):
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        return comment

@app.route('/')
def home():
    return "Welcome to our Flask API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments or not isinstance(comments, list):
        return jsonify({"error": "comments must be a list of strings"}), 400

    try:
        preprocessed_comments = [preprocess_comment(c) for c in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return jsonify(response)

if __name__ == '__main__':
    # Disable reloader to avoid loading the model twice in debug mode
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
