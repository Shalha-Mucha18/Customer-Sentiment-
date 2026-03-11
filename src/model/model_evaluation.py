import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import json
import mlflow.sklearn
import dagshub
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
from src.logger import logging

MODEL_ARTIFACT_PATH = "model"

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Shalha-Mucha18"
repo_name = "Capstone-Project-2026"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')



# # Below code block is for  local use
# # -------------------------------------------------------------------------------------

# mlflow.set_tracking_uri('https://dagshub.com/Shalha-Mucha18/Capstone-Project-2026.mlflow')
# dagshub.init(repo_owner='Shalha-Mucha18', repo_name='Capstone-Project-2026', mlflow=True)
# # -------------------------------------------------------------------------------------


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file)
        logging.info('Evaluation metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving metrics to file: %s', e)
        raise

def save_model_info(run_id: str, model_uri: str, file_path: str):
    """Save the run ID and model URI to a JSON file."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        model_info = {'run_id': run_id, 'model_uri': model_uri}
        with open(file_path, 'w') as file:
            json.dump(model_info, file)
        logging.info('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error saving model info to file: %s', e)
        raise

def main():

    mlflow.set_experiment("Model Evaluation")
    with  mlflow.start_run() as run:

        try:
            clf = load_model('models/model.pkl')
            test_data = load_data('data/processed/test_bow.csv')
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values
            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')

            # Log metrics to MLflow

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log the model to MLflow and keep its URI for registration.
            logged_model_info = mlflow.sklearn.log_model(
                sk_model=clf,
                name=MODEL_ARTIFACT_PATH,
            )

            # Save model info to a JSON file    
            save_model_info(
                run.info.run_id,
                logged_model_info.model_uri,
                'reports/model_info.json',
            )
        except Exception as e:
            logging.error('Error in main function: %s', e)
            raise

if __name__ == "__main__":
    main()
