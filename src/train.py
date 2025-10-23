"""
Module d'entraînement du modèle de prédiction de churn.
Intègre MLflow pour le tracking des expériences.
"""
import os
import json
import joblib
import socket
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from datetime import datetime

from src.data_preprocessing import load_data, make_preprocessor, split_data
from src.utils import MODEL_PATH, METRICS_PATH, DATA_PATH, EXPERIMENT_NAME


# ============================================
# DÉTECTION AUTOMATIQUE DE L'ENVIRONNEMENT
# ============================================
def get_mlflow_tracking_uri():
    """
    Détecte automatiquement l'environnement et retourne la bonne URI MLflow.
    - Dans Docker (Airflow) : http://host.docker.internal:5000
    - En local : http://localhost:5000
    """
    hostname = socket.gethostname()
    is_docker = (
        hostname.startswith(('airflow', 'mlops')) or 
        os.path.exists('/.dockerenv')
    )
    
    if is_docker:
        uri = "http://host.docker.internal:5000"
        print(f"🐳 Environnement Docker détecté → MLflow URI: {uri}")
    else:
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        print(f"💻 Environnement Local détecté → MLflow URI: {uri}")
    
    return uri


# Configuration MLflow avec détection automatique
MLFLOW_TRACKING_URI = get_mlflow_tracking_uri()
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI


def train_model(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42):
    """
    Entraîne le modèle de prédiction de churn avec tracking MLflow.
    
    Args:
        n_estimators: Nombre d'arbres dans la forêt
        max_depth: Profondeur maximale des arbres
        min_samples_split: Nombre minimum d'échantillons pour split
        random_state: Seed pour reproductibilité
    
    Returns:
        Pipeline entraîné et métriques
    """
    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Chargement et préparation des données
    print("📊 Chargement des données...")
    df = load_data(str(DATA_PATH))
    print(f"✅ {len(df)} lignes chargées")
    
    X, y, preprocessor = make_preprocessor(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"🔀 Split: {len(X_train)} train / {len(X_test)} test")
    
    # Démarrage du run MLflow
    with mlflow.start_run(run_name=f"churn_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log des paramètres
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "random_state": random_state,
            "test_size": 0.2
        }
        mlflow.log_params(params)
        
        # Création et entraînement du modèle
        print("🎯 Entraînement du modèle...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Prédictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calcul des métriques
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred))
        }
        
        # Log des métriques dans MLflow
        mlflow.log_metrics(metrics)
        
        # Sauvegarde du modèle dans MLflow
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name="churn_predictor"
        )
        
        # Sauvegarde locale
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        # Affichage des résultats
        print("\n" + "="*50)
        print("📈 RÉSULTATS D'ENTRAÎNEMENT")
        print("="*50)
        for metric_name, value in metrics.items():
            print(f"{metric_name:15s}: {value:.4f}")
        print("="*50)
        print(f"✅ Modèle sauvegardé: {MODEL_PATH}")
        print(f"✅ Métriques sauvegardées: {METRICS_PATH}")
        
        return pipeline, metrics


def main():
    """Point d'entrée principal."""
    try:
        pipeline, metrics = train_model(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        print("\n🎉 Entraînement terminé avec succès!")
        return 0
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {str(e)}")
        raise


if __name__ == "__main__":
    main()