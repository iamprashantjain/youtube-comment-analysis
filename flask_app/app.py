import os
import yaml
import sys
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import logging
from datetime import datetime


# Logging Setup
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)
LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

# Basic file logging
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILEPATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)
logger = logging.getLogger(__name__)

# Custom Exception
class customexception(Exception):
    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        try:
            _, _, exc_tb = error_details.exc_info()
            self.lineno = exc_tb.tb_lineno if exc_tb else -1
            self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "<unknown>"
        except Exception:
            # Fallback if exc_info is not available
            self.lineno = -1
            self.file_name = "<unknown>"

    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(self.error_message)
        )

# Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# NLTK safety net
def _ensure_nltk_resources():
    """
    Ensure required NLTK corpora are available. Download if missing.
    """
    try:
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("NLTK stopwords not found. Downloading...")
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("NLTK wordnet not found. Downloading...")
            nltk.download('wordnet')
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            # Not strictly necessary here but commonly used with wordnet/lemmatizer stacks
            logger.info("NLTK punkt not found. Downloading...")
            nltk.download('punkt')
    except Exception as e:
        # Don't crash the app if NLTK download fails; just log it.
        logger.error("Failed to ensure NLTK resources: %s", e)

_ensure_nltk_resources()


# Preprocessing
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        if not isinstance(comment, str):
            comment = str(comment)

        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        try:
            sw = set(stopwords.words('english'))
        except LookupError:
            _ensure_nltk_resources()
            sw = set(stopwords.words('english'))
        sw = sw - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in sw])

        # Lemmatize the words
        try:
            lemmatizer = WordNetLemmatizer()
        except Exception:
            _ensure_nltk_resources()
            lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error("Error in preprocessing comment: %s", e)
        # Return original to avoid hard failure
        return comment

# Model Loading
def load_model_and_vectorizer():
    try:
        # Load parameters
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        logger.info("Successfully loaded params.yaml")

        # Configure MLflow tracking
        mlflow_params = params["mlflow"]

        # Get token from environment variable
        token = os.environ.get("DAGSHUB_PAT")
        if not token:
            raise ValueError("DAGSHUB_PAT environment variable not found")

        # Use environment token instead of mlflow_params['token']
        mlflow.set_tracking_uri(
            f"https://{mlflow_params['username']}:{token}@dagshub.com/{mlflow_params['repo']}"
        )

        logger.info("Configured MLflow tracking URI for %s", mlflow_params['repo'])

        # Load model from MLflow registry
        model_uri = "models:/youtube_chromeplugin_model/2"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Successfully loaded model from MLflow registry: %s", model_uri)

        # Load vectorizer from local file
        vectorizer_path = "artifacts/model_trainer/tfidf_vectorizer.pkl"
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Successfully loaded vectorizer from %s", vectorizer_path)

        return model, vectorizer

    except Exception as e:
        logger.error("Error loading model/vectorizer: %s", str(e))
        logger.error(traceback.format_exc())
        raise customexception(e, sys)

# Global load
try:
    model, vectorizer = load_model_and_vectorizer()
except customexception as ce:
    # Fail-fast if model can't be loaded
    logger.critical(str(ce))
    raise

# Routes
@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    try:
        data = request.json
        if data is None:
            raise customexception("Invalid JSON payload", sys)

        comments_data = data.get('comments')
        if not comments_data:
            return jsonify({"error": "No comments provided"}), 400

        comments = [item.get('text', '') for item in comments_data]
        timestamps = [item.get('timestamp', None) for item in comments_data]

        logger.info("Received %d comments for prediction_with_timestamps", len(comments))

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]

        # Return the response with original comments, predicted sentiments, and timestamps
        response = [
            {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
            for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
        ]

        logger.info("Prediction completed for predict_with_timestamps")
        return jsonify(response)

    except customexception as ce:
        logger.error(str(ce))
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        ce = customexception(e, sys)
        logger.error(str(ce))
        return jsonify({"error": str(ce)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if data is None:
            raise customexception("Invalid JSON payload", sys)

        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        logger.info("Received %d comments for prediction", len(comments))

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]

        # Return the response with original comments and predicted sentiments
        response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]

        logger.info("Prediction completed for predict")
        return jsonify(response)

    except customexception as ce:
        logger.error(str(ce))
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        ce = customexception(e, sys)
        logger.error(str(ce))
        return jsonify({"error": str(ce)}), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        if data is None:
            raise customexception("Invalid JSON payload", sys)

        sentiment_counts = data.get('sentiment_counts')
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        logger.info("Generating pie chart for sentiment counts: %s", sentiment_counts)

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise customexception("Sentiment counts sum to zero", sys)

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        logger.info("Pie chart generated successfully")
        return send_file(img_io, mimetype='image/png')
    except customexception as ce:
        logger.error(str(ce))
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        ce = customexception(e, sys)
        logger.error("Error in /generate_chart: %s", str(ce))
        return jsonify({"error": str(ce)}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        if data is None:
            raise customexception("Invalid JSON payload", sys)

        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        logger.info("Generating wordcloud for %d comments", len(comments))

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        # NOTE: We keep your configuration intact.
        try:
            sw = set(stopwords.words('english'))
        except LookupError:
            _ensure_nltk_resources()
            sw = set(stopwords.words('english'))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=sw,
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        logger.info("Wordcloud generated successfully")
        return send_file(img_io, mimetype='image/png')
    except customexception as ce:
        logger.error(str(ce))
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        ce = customexception(e, sys)
        logger.error("Error in /generate_wordcloud: %s", str(ce))
        return jsonify({"error": str(ce)}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        if data is None:
            raise customexception("Invalid JSON payload", sys)

        sentiment_data = data.get('sentiment_data')
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        logger.info("Generating trend graph for %d records", len(sentiment_data))

        df = pd.DataFrame(sentiment_data)

        # Parse timestamps safely
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        # Drop rows with invalid timestamps to avoid plotting errors
        before = len(df)
        df = df.dropna(subset=['timestamp'])
        after = len(df)
        if after == 0:
            raise customexception("All timestamps are invalid or missing", sys)
        if before != after:
            logger.warning("Dropped %d rows with invalid timestamps", before - after)

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Make sure sentiment is numeric
        df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['sentiment'])
        if df.empty:
            raise customexception("No valid sentiment values to plot", sys)
        df['sentiment'] = df['sentiment'].astype(int)

        # Debug: Check unique dates and sentiments
        logger.info("Unique days: %s", str(df.index.normalize().unique()))
        logger.info("Sentiment counts: %s", str(df['sentiment'].value_counts().to_dict()))

        # Resample daily instead of monthly
        daily_counts = df.resample('D')['sentiment'].value_counts().unstack(fill_value=0)
        daily_totals = daily_counts.sum(axis=1)
        # Avoid divide-by-zero
        daily_totals = daily_totals.replace(0, np.nan)
        daily_percentages = (daily_counts.T / daily_totals).T * 100
        daily_percentages = daily_percentages.fillna(0)

        # Ensure all sentiment columns exist
        for val in [-1, 0, 1]:
            if val not in daily_percentages.columns:
                daily_percentages[val] = 0

        daily_percentages = daily_percentages[[-1, 0, 1]]

        # Check if there’s data to plot
        if daily_percentages.empty:
            return jsonify({"error": "Not enough data to plot trend"}), 400

        # Plot
        plt.figure(figsize=(12, 6))
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        for val in [-1, 0, 1]:
            plt.plot(
                daily_percentages.index,
                daily_percentages[val],
                marker='o',
                linestyle='-',
                label=sentiment_labels[val],
                color=colors[val]
            )

        plt.title('Daily Sentiment Trend')
        plt.xlabel('Date')
        plt.ylabel('Percentage of Comments')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        logger.info("Trend graph generated successfully")
        return send_file(img_io, mimetype='image/png')
    except customexception as ce:
        logger.error(str(ce))
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        ce = customexception(e, sys)
        logger.error("Error in /generate_trend_graph: %s", str(ce))
        return jsonify({"error": str(ce)}), 500


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)     # for github action runner, debug should be false
