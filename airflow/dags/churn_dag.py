from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {"owner": "airflow", "retries": 0}

with DAG(
    dag_id="churn_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # déclenchement manuel
    catchup=False,
    default_args=default_args,
    tags=["mlops", "churn"],
):
    train = BashOperator(
        task_id="train_model",
        bash_command=(
            # installe les deps dans le conteneur Airflow (démo)
            "pip install --no-cache-dir -r /opt/project/requirements.txt && "
            # exécute ton code monté depuis l'hôte
            "python -m src.train"
        ),
        env={"PYTHONPATH": "/opt/project"},  # pour que 'src' soit importable
    )

    train
