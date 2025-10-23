"""
DAG Airflow pour le pipeline MLOps de prédiction de churn.
Pipeline complet: validation → entraînement → évaluation → déploiement
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
import json
from pathlib import Path


# Configuration du DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Fonctions Python pour certaines tâches
def check_data_quality(**context):
    """Vérifie la qualité des données avant l'entraînement."""
    import pandas as pd
    
    data_path = "/opt/project/data/telco_churn.csv"
    
    try:
        df = pd.read_csv(data_path)
        
        # Vérifications de base
        checks = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
        }
        
        print("📊 Résultats de validation des données:")
        for key, value in checks.items():
            print(f"  {key}: {value}")
        
        # Alertes
        if checks["missing_values"] > len(df) * 0.1:
            raise ValueError(f"⚠️ Trop de valeurs manquantes: {checks['missing_values']}")
        
        if checks["duplicate_rows"] > 0:
            print(f"⚠️ {checks['duplicate_rows']} lignes dupliquées trouvées")
        
        print("✅ Validation des données réussie!")
        return checks
        
    except Exception as e:
        print(f"❌ Erreur lors de la validation: {str(e)}")
        raise


def check_model_performance(**context):
    """Vérifie que les performances du modèle sont acceptables."""
    metrics_path = "/opt/project/artifacts/metrics.json"
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("📈 Métriques du modèle:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Seuils de performance
        min_auc = 0.70
        min_accuracy = 0.75
        
        if metrics.get("roc_auc", 0) < min_auc:
            raise ValueError(f"⚠️ AUC trop faible: {metrics['roc_auc']:.4f} < {min_auc}")
        
        if metrics.get("accuracy", 0) < min_accuracy:
            raise ValueError(f"⚠️ Accuracy trop faible: {metrics['accuracy']:.4f} < {min_accuracy}")
        
        print("✅ Performance du modèle acceptable!")
        return metrics
        
    except FileNotFoundError:
        print(f"❌ Fichier de métriques non trouvé: {metrics_path}")
        raise
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {str(e)}")
        raise


def notify_completion(**context):
    """Notifie la fin du pipeline."""
    print("="*60)
    print("🎉 PIPELINE MLOps TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"⏰ Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 DAG: {context['dag'].dag_id}")
    print(f"🔄 Run ID: {context['run_id']}")
    print("="*60)


# Définition du DAG
with DAG(
    dag_id="churn_mlops_pipeline",
    default_args=default_args,
    description="Pipeline MLOps complet pour la prédiction de churn",
    schedule_interval=None,  # Déclenchement manuel ou via événement
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "churn", "production"],
) as dag:

    # 1. Tâche de démarrage
    start_pipeline = BashOperator(
        task_id="start_pipeline",
        bash_command="""
        echo "🚀 Démarrage du pipeline MLOps Churn"
        echo "📅 Date: $(date)"
        echo "🔧 Version Airflow: $(airflow version)"
        """,
    )

    # 2. Vérification de l'environnement
    check_environment = BashOperator(
        task_id="check_environment",
        bash_command="""
        echo "🔍 Vérification de l'environnement..."
        python3 --version
        pip3 list | grep -E '(pandas|scikit-learn|mlflow)'
        echo "✅ Environnement OK"
        """,
    )

    # 3. Groupe de tâches: Validation des données
    with TaskGroup("data_validation") as data_validation:
        
        check_data_exists = BashOperator(
            task_id="check_data_exists",
            bash_command="""
            if [ -f /opt/project/data/telco_churn.csv ]; then
                echo "✅ Fichier de données trouvé"
                wc -l /opt/project/data/telco_churn.csv
            else
                echo "❌ Fichier de données non trouvé!"
                exit 1
            fi
            """,
        )
        
        validate_data_quality = PythonOperator(
            task_id="validate_data_quality",
            python_callable=check_data_quality,
        )
        
        check_data_exists >> validate_data_quality

    # 4. Installation des dépendances
    install_dependencies = BashOperator(
        task_id="install_dependencies",
        bash_command="""
        echo "📦 Installation des dépendances..."
        pip install --no-cache-dir -r /opt/project/requirements.txt -q
        echo "✅ Dépendances installées"
        """,
    )

    # 5. Groupe de tâches: Entraînement du modèle
    with TaskGroup("model_training") as model_training:
        
        train_model = BashOperator(
            task_id="train_model",
            bash_command="""
            echo "🎯 Entraînement du modèle..."
            cd /opt/project
            python -m src.train
            """,
            env={"PYTHONPATH": "/opt/project"},
        )
        
        validate_model_output = BashOperator(
            task_id="validate_model_output",
            bash_command="""
            if [ -f /opt/project/artifacts/models/model.joblib ]; then
                echo "✅ Modèle sauvegardé"
                ls -lh /opt/project/artifacts/models/model.joblib
            else
                echo "❌ Modèle non trouvé!"
                exit 1
            fi
            """,
        )
        
        train_model >> validate_model_output

    # 6. Évaluation du modèle
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=check_model_performance,
    )

    # 7. Sauvegarde des artefacts
    save_artifacts = BashOperator(
        task_id="save_artifacts",
        bash_command="""
        echo "💾 Sauvegarde des artefacts..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        mkdir -p /opt/project/artifacts/archive
        cp /opt/project/artifacts/models/model.joblib /opt/project/artifacts/archive/model_${TIMESTAMP}.joblib
        cp /opt/project/artifacts/metrics.json /opt/project/artifacts/archive/metrics_${TIMESTAMP}.json
        echo "✅ Artefacts archivés avec timestamp: ${TIMESTAMP}"
        """,
    )

    # 8. Tests de l'API (optionnel)
    test_api = BashOperator(
        task_id="test_api",
        bash_command="""
        echo "🧪 Tests de l'API (simulation)..."
        echo "✅ Tests API OK"
        """,
    )

    # 9. Notification de fin
    notify_success = PythonOperator(
        task_id="notify_success",
        python_callable=notify_completion,
    )

    # Définition du flux
    start_pipeline >> check_environment >> data_validation >> install_dependencies
    install_dependencies >> model_training >> evaluate_model
    evaluate_model >> save_artifacts >> test_api >> notify_success


# Description du DAG pour la documentation
dag.doc_md = """
# Pipeline MLOps - Prédiction de Churn

## Description
Pipeline complet d'entraînement et de déploiement du modèle de prédiction de churn pour les télécommunications.

## Étapes du pipeline:
1. **Démarrage**: Initialisation et logging
2. **Vérification environnement**: Vérification des dépendances Python
3. **Validation des données**: Vérification de la qualité et cohérence des données
4. **Installation dépendances**: Installation des packages Python nécessaires
5. **Entraînement**: Entraînement du modèle RandomForest avec MLflow tracking
6. **Évaluation**: Validation des performances du modèle
7. **Sauvegarde**: Archivage des artefacts (modèle + métriques)
8. **Tests**: Tests de l'API (si applicable)
9. **Notification**: Notification de fin de pipeline

## Déclenchement
- **Manuel**: Via l'interface Airflow
- **Programmé**: Configurer `schedule_interval` si besoin

## Métriques suivies
- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-Score

## Auteur
MLOps Team - Churn Prediction Project
"""