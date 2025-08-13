import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import multiprocessing
import warnings
import scipy.sparse

warnings.filterwarnings("ignore")

# ========== MLflow Setup ==========
username = "iamprashantjain"
token = "7bed6b5be2021b1a4eaae221787bcb048ab2bcfd"
mlflow.set_tracking_uri(
    f"https://{username}:{token}@dagshub.com/{username}/youtube-comment-analysis.mlflow"
)
mlflow.set_experiment("All Algorithms Comparison")

# ========== Data Loading ==========
df = pd.read_csv("preprocessed_data.csv").dropna(subset=['clean_comment'])
df = df.dropna(subset=['clean_comment', 'category'])
df['clean_comment'] = df['clean_comment'].astype(str).str.strip()
df = df[df['clean_comment'] != ""]
df = df[df['category'].isin([-1, 0, 1])]
df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1}).astype(int)

# ========== Vectorization ==========
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000)
X = vectorizer.fit_transform(df['clean_comment'])
y = df['category']

# Ensure matrix is writable (avoid WRITEBACKIFCOPY issues)
X = X.tocsr()
X.sort_indices()

# ========== SMOTE & Split ==========
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ========== MLflow Logging ==========
def log_mlflow(model_name, model, X_train_local, X_test_local, y_train_local, y_test_local, final_params):
    """
    Starts its own MLflow run and logs params, metrics and model artifact.
    Using local copies of data to avoid cross-process issues.
    """
    with mlflow.start_run(run_name=f"{model_name}_SMOTE_TFIDF_Trigrams"):
        mlflow.set_tag('experiment_type', 'algorithm_comparison')
        mlflow.log_param('model_name', model_name)

        # log params
        for param_name, param_value in final_params.items():
            try:
                mlflow.log_param(str(param_name), str(param_value))
            except Exception:
                # ensure robust logging
                mlflow.log_param(str(param_name), repr(param_value))

        # fit and evaluate
        model.fit(X_train_local, y_train_local)
        y_pred = model.predict(X_test_local)

        accuracy = accuracy_score(y_test_local, y_pred)
        mlflow.log_metric("accuracy", float(accuracy))

        classification_rep = classification_report(y_test_local, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    # label might be like '0', '1', 'macro avg', etc.
                    mlflow.log_metric(f"{label}_{metric}".replace(" ", "_"), float(value))

        # artifact
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        return accuracy

# ========== Hyperparameter Spaces ==========
param_spaces = {
    "RandomForest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 50, 300),
        "max_depth": t.suggest_int("max_depth", 5, 30),
    },
    "LogisticRegression": lambda t: {
        "C": t.suggest_float("C", 1e-3, 1e2, log=True),
    },
    # SVM tuned as LinearSVC (fast for text)
    "SVM": lambda t: {
        "C": t.suggest_float("C", 1e-3, 1e2, log=True),
    },
    "NaiveBayes": lambda t: {
        "alpha": t.suggest_float("alpha", 0.0, 1.0),
    },
    "KNN": lambda t: {
        "n_neighbors": t.suggest_int("n_neighbors", 3, 15),
    },
    "XGBoost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 50, 300),
        "max_depth": t.suggest_int("max_depth", 3, 10),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3),
    },
    "LightGBM": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 50, 300),
        "max_depth": t.suggest_int("max_depth", 3, 10),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3),
    },
}

# ========== Model Mapping ==========
models = {
    "RandomForest": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVM": LinearSVC,  # use LinearSVC for speed on text
    "NaiveBayes": MultinomialNB,
    "KNN": KNeighborsClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
}

# Defaults/constants to ensure final model instantiation has necessary fixed args
param_defaults = {
    "RandomForest": {"random_state": 42},
    "LogisticRegression": {"max_iter": 500, "random_state": 42, "solver": "lbfgs"},
    "SVM": {"max_iter": 2000, "dual": False, "random_state": 42},
    "NaiveBayes": {},
    "KNN": {},
    "XGBoost": {"random_state": 42, "use_label_encoder": False, "eval_metric": "mlogloss"},
    "LightGBM": {"random_state": 42},
}

# ========== Objective Function ==========
def objective(trial, model_name, X_train_local, y_train_local, X_test_local, y_test_local):
    """
    Builds params from trial, instantiates model and returns accuracy.
    Uses copies for SVM to avoid SciPy WRITEBACKIFCOPY issues.
    """
    params = param_spaces[model_name](trial)
    # combine with defaults for fit-time behavior if needed (but trial params should override)
    combined = {**param_defaults.get(model_name, {}), **params}
    model = models[model_name](**combined)

    # SVM / LinearSVC: ensure working on a fresh copy to avoid sparse in-place issues
    if model_name == "SVM":
        X_train_used = X_train_local.copy()
        X_test_used = X_test_local.copy()
    else:
        X_train_used = X_train_local
        X_test_used = X_test_local

    model.fit(X_train_used, y_train_local)
    preds = model.predict(X_test_used)
    return accuracy_score(y_test_local, preds)

# ========== Single Model Run (sequential, robust MLflow logging) ==========
def run_for_one_model(model_name, n_trials=30):
    print(f"\nRunning Optuna for {model_name} with n_trials={n_trials}...")

    # create local copies for safety
    X_train_local = X_train.copy()
    X_test_local = X_test.copy()
    y_train_local = y_train.copy()
    y_test_local = y_test.copy()

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, model_name, X_train_local, y_train_local, X_test_local, y_test_local),
        n_trials=n_trials
    )

    # Get best params (dict) and merge with defaults so constants are present
    best_params = study.best_params.copy()
    final_params = {**param_defaults.get(model_name, {}), **best_params}

    # instantiate final model with merged params
    final_model = models[model_name](**final_params)

    # log to MLflow (this starts its own run)
    log_mlflow(model_name, final_model, X_train_local, X_test_local, y_train_local, y_test_local, final_params)

# ========== Main (sequential to ensure reliable MLflow runs) ==========
def run_optune_for_all():
    # Run SVM with fewer trials (fast LinearSVC)
    run_for_one_model("SVM", n_trials=5)

    # Remaining models: run sequentially (reliable MLflow logging)
    other_models = [m for m in models.keys() if m != "SVM"]
    for m in other_models:
        # You can tune trials per model here (default 30). Keep lower if memory/CPU constrained.
        run_for_one_model(m, n_trials=30)

if __name__ == "__main__":
    run_optune_for_all()
