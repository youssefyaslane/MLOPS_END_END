"""
Utilitaires et configuration du projet MLOps Churn.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration des chemins
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = ARTIFACTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = DATA_DIR / "telco_churn.csv"

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_prediction")

# Configuration API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Configuration du modèle
MODEL_NAME = os.getenv("MODEL_NAME", "churn_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

# Configuration de l'environnement
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
PROJECT_NAME = os.getenv("PROJECT_NAME", "MLOPS-CHURN-TELECOM")


def get_project_info():
    """Retourne les informations du projet."""
    return {
        "project_name": PROJECT_NAME,
        "environment": ENVIRONMENT,
        "model_path": str(MODEL_PATH),
        "data_path": str(DATA_PATH),
        "mlflow_uri": MLFLOW_TRACKING_URI
    }


if __name__ == "__main__":
    print("📁 Configuration du projet:")
    for key, value in get_project_info().items():
        print(f"  {key}: {value}")