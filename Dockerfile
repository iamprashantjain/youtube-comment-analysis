FROM python:3.10-slim

WORKDIR /app

# Install required system packages -- "libgomp1" provides parallel computing for lightgbm 
RUN apt-get update && apt-get install -y libgomp1

# Copy the entire flask app and artifacts to preserve structure
COPY flask_app/ /app/
COPY artifacts/model_trainer/tfidf_vectorizer.pkl /app/artifacts/model_trainer/tfidf_vectorizer.pkl
COPY params.yaml /app/params.yaml

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download necessary NLTK data
RUN python -m nltk.downloader stopwords wordnet

# Expose the port Flask will run on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]