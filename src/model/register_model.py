import json 
import mlflow
import logging
import dagshub
import os
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
from src.logger import logging  

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


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
# -------------------------------------------------------------------------------------





# # Below code block is for  local use
# # -------------------------------------------------------------------------------------

# mlflow.set_tracking_uri('https://dagshub.com/Shalha-Mucha18/Capstone-Project-2026.mlflow')
# dagshub.init(repo_owner='Shalha-Mucha18', repo_name='Capstone-Project-2026', mlflow=True)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = model_info.get("model_uri")
        run_id = model_info.get("run_id")

        # Preferred path for MLflow 3.x: discover logged model by run_id.
        if not model_uri and run_id:
            client = mlflow.tracking.MlflowClient()
            logged_models = client.search_logged_models(
                filter_string=f"source_run_id='{run_id}'",
                max_results=1,
            )
            if logged_models:
                model_id = logged_models[0].model_id
                model_uri = f"models:/{model_id}"
                logging.info("Resolved logged model URI from run_id: %s", model_uri)

        # Backward-compatibility for older metadata that stores run_id/model_path.
        if not model_uri and run_id:
            model_path = model_info.get("model_path", "model")
            if model_path == "modelsl":
                logging.warning("Detected legacy model_path='modelsl'; using 'model' instead.")
                model_path = "model"
            model_uri = f"runs:/{run_id}/{model_path}"            

        if not model_uri:
            raise ValueError("model_uri missing and unable to resolve from run metadata")

        logging.info("Registering model from URI: %s", model_uri)
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logging.info(
            "Model %s version %s registered and transitioned to Staging.",
            model_name,
            model_version.version,
        )
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/model_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
     
        
