#!/bin/bash
set -e

# Script pour créer plusieurs bases de données dans PostgreSQL
# Utilisé par docker-compose pour initialiser les BDD Airflow et MLflow

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Base de données pour MLflow
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;
    
    -- Base de données pour Airflow (déjà créée par défaut)
    GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
EOSQL

echo "✅ Bases de données créées: airflow, mlflow"