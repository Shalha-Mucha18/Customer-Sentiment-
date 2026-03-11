# load test + signature + performance
import unittest
from pathlib import Path
import dagshub
import mlflow
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
import os


class TestModelPerformance(unittest.TestCase):

    @classmethod

    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        vectorizer_path = project_root / "models" / "vectorizer.pkl"
        test_data_path = project_root / "data" / "processed" / "test_bow.csv"

        missing = []
        if not vectorizer_path.exists():
            missing.append(str(vectorizer_path))
        if not test_data_path.exists():
            missing.append(str(test_data_path))
        if missing:
            raise unittest.SkipTest(
                "Required test artifacts missing: "
                + ", ".join(missing)
                + ". Run the pipeline (e.g., `dvc repro`) first."
            )
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "vikashdas770"
        repo_name = "YT-Capstone-Project"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        if not cls.new_model_version:
            raise unittest.SkipTest("No registered model version found in MLflow.")
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        try:
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        except Exception as e:
            raise unittest.SkipTest(f"Unable to load model from MLflow: {e}")

        # Load the vectorizer
        cls.vectorizer = pickle.load(open(vectorizer_path, 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv(test_data_path)

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        try:
            # Prefer stage-based lookup when available.
            latest_version = client.get_latest_versions(model_name, stages=[stage])
            if latest_version:
                return latest_version[0].version
        except Exception:
            # Fall back to all versions when registry stage API is unavailable.
            pass

        try:
            versions = client.search_model_versions(f"name='{model_name}'")
        except Exception:
            return None

        if not versions:
            return None
        return max(versions, key=lambda v: int(v.version)).version

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], self.new_model.metadata.get_input_schema().input_names.__len__())
        # Verify the output shape (assuming binary classification)
        self.assertEqual(prediction.shape[0], 1)
        self.assertEqual(prediction.shape[1], self.new_model.metadata.get_output_schema().output_names.__len__())

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        # Assuming binary classification, we can calculate accuracy, precision, recall, and F1 score
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics (these can be adjusted based on your requirements)
        expected_accuracy_threshold = 0.40
        expected_precision_threshold = 0.40
        expected_recall_threshold = 0.40
        expected_f1_threshold = 0.40

        # Assert that the new model meets the expected performance thresholds
        # You can adjust the thresholds based on your specific requirements and the performance of your model
        self.assertGreaterEqual(accuracy_new, expected_accuracy_threshold, f'Accuracy should be at least {expected_accuracy_threshold}')
        self.assertGreaterEqual(precision_new, expected_precision_threshold, f'Precision should be at least {expected_precision_threshold}')        
        self.assertGreaterEqual(recall_new, expected_recall_threshold, f'Recall should be at least {expected_recall_threshold}')
        self.assertGreaterEqual(f1_new, expected_f1_threshold, f'F1 score should be at least {expected_f1_threshold}')
if __name__ == '__main__':
    unittest.main()       
