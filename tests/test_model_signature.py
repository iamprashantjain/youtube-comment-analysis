import os
import yaml
import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient


def setup_mlflow():
    """Setup MLflow tracking with Dagshub credentials."""
    with open("params.yaml", "r") as f:
        mlflow_params = yaml.safe_load(f)["mlflow"]

    token = os.environ.get("DAGSHUB_PAT")
    if not token:
        raise ValueError("DAGSHUB_PAT environment variable not found")

    tracking_uri = (
        f"https://{mlflow_params['username']}:{token}"
        f"@dagshub.com/{mlflow_params['repo']}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("youtube_chromeplugin_model", "Staging", os.path.join("artifacts", "model_trainer", "tfidf_vectorizer.pkl")),
])
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    client = setup_mlflow()

    try:
        # Load model directly by stage
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Load vectorizer
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)

        # Dummy input
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # Run prediction
        prediction = model.predict(input_df)

        # Assertions
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), \
            "Input feature count mismatch"
        assert len(prediction) == input_df.shape[0], \
            "Output row count mismatch"

        print(f"Model '{model_name}' in stage '{stage}' successfully processed input.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")
