# 🎯 MLOps End-to-End : Prédiction du Churn Client - Télécommunications

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-2.9.2-green?logo=mlflow)
![Airflow](https://img.shields.io/badge/Airflow-2.9.3-red?logo=apache-airflow)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Projet MLOps complet** : Pipeline end-to-end de prédiction du churn avec tracking MLflow, orchestration Airflow, déploiement Docker et API REST FastAPI.

## 📊 Vue d'ensemble

Ce projet implémente un **pipeline MLOps complet et production-ready** pour prédire le churn (désabonnement) des clients dans le secteur des télécommunications. Il suit les meilleures pratiques MLOps en intégrant :

- 🔬 **MLflow** : Tracking des expériences, versioning et registry des modèles
- 🌊 **Airflow** : Orchestration automatisée du pipeline de ML
- 🐳 **Docker** : Containerisation de tous les services
- 🚀 **FastAPI** : API REST pour servir les prédictions en temps réel
- 🧪 **Tests automatisés** : Pytest pour validation continue
- 📊 **Preprocessing complet** : Pipeline de transformation des données

### 🎯 Résultats Obtenus

| Métrique | Score | Description |
|----------|-------|-------------|
| **ROC-AUC** | **82.29%** | Excellente capacité de discrimination |
| **Accuracy** | **79.03%** | Bonne précision globale |
| **Precision** | **62.95%** | Taux de vrais positifs parmi les prédictions positives |
| **Recall** | **51.34%** | Taux de clients churners détectés |
| **F1-Score** | **56.55%** | Moyenne harmonique precision/recall |

---

## 🏗️ Architecture

### Architecture Globale

```
┌─────────────────────────────────────────────────────────────────┐
│                     STACK MLOPS COMPLÈTE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   MLflow     │    │   Airflow    │    │   FastAPI    │     │
│  │  Tracking    │◄───┤ Orchestration│◄───┤  REST API    │     │
│  │  (Port 5000) │    │  (Port 8080) │    │ (Port 8000)  │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                    │                    │            │
│         │                    │                    │            │
│  ┌──────▼────────────────────▼────────────────────▼──────┐    │
│  │              Docker Network (mlops-network)            │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                 │
│                    ┌─────────▼─────────┐                       │
│                    │   PostgreSQL      │                       │
│                    │  Database Airflow │                       │
│                    │   (Port 5432)     │                       │
│                    └───────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

                              │
                    ┌─────────▼─────────┐
                    │  Volumes Docker   │
                    │  - mlflow-artifacts│
                    │  - postgres-data   │
                    │  - airflow-logs    │
                    └───────────────────┘
```

### Architecture du Pipeline ML

```
┌─────────────────────────────────────────────────────────────┐
│                   PIPELINE MACHINE LEARNING                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 DATA          →  🔄 PREPROCESSING  →  🎯 TRAINING      │
│  telco_churn.csv     - One-Hot Encoding    - Random Forest  │
│  (7032 samples)      - StandardScaler      - Hyperparams    │
│                      - Feature Engineering                   │
│                                                             │
│           ↓                                                 │
│                                                             │
│  📈 EVALUATION    →  💾 SAVING         →  🚀 DEPLOYMENT    │
│  - ROC-AUC           - Joblib              - FastAPI        │
│  - Accuracy          - MLflow Registry     - Docker         │
│  - Metrics JSON                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
---

## 📁 Structure du Projet

```
mlops-churn-telecom/
│
├── 📁 FICHIERS DE CONFIGURATION/
│   ├── .env.example              # Template de configuration (sans secrets)
│   
├── 📁 airflow/                   # Orchestration Airflow
│   ├── dags/
│   │   ├── churn_dag.py          # DAG simple (version 1)
│   │   └── churn_dag_improved.py # DAG complet avec 9 étapes (version finale)
│   └── home/                     # Configuration Airflow
│       ├── airflow.cfg           # Config principale
│       └── webserver_config.py   # Config serveur web
│
├── 📁 docker/                    # Infrastructure Docker
│   ├── docker-compose.yml        # Orchestration de tous les services
│   ├── docker-compose-final.yml  # Configuration complète
│   ├── Dockerfile.api            # Image Docker pour l'API FastAPI
│   └── scripts/
│       └── init-postgres.sh      # Initialisation PostgreSQL
│
├── 📁 src/                       # Code source Python
│   ├── __init__.py               # Package Python
│   ├── api.py                    # API FastAPI (endpoints, validation)
│   ├── train.py                  # Script d'entraînement avec MLflow
│   ├── data_preprocessing.py     # Pipeline de preprocessing sklearn
│   └── utils.py                  # Utilitaires (chemins, config, constantes)
│
├── 📁 tests/                     # Tests automatisés
│   ├── __init__.py               # Package de tests
│   ├── test_api.py               # Tests de l'API (pytest)
│   ├── test_data.py              # Tests des données
│   └── test_preprocessing.py     # Tests du preprocessing
│
├── 📁 data/                      # Données
│   └── telco_churn.csv           # Dataset (7032 clients)
│
├── 📁 notebooks/                 # Analyse exploratoire (optionnel)
│   └── exploratory_analysis.ipynb # EDA, visualisations
│
├── 📁 artifacts/                 # Artefacts générés (Git-ignored)
│   ├── models/
│   │   └── model.joblib          # Modèle entraîné
│   └── metrics.json              # Métriques du dernier run
│
├── 📁 mlruns/                    # Tracking MLflow local (Git-ignored)
│   └── ...                       # Expériences et runs
│
├── 📁 venv/                      # Environnement virtuel (Git-ignored)
│
├── 📄 .gitignore                 # Fichiers à ignorer par Git
├── 📄 README.md                  # Ce fichier !
├── 📄 requirements.txt           # Dépendances Python
```

### 📝 Description des Fichiers Clés

| Fichier | Description |
|---------|-------------|
| `src/train.py` | Script principal d'entraînement avec détection auto Docker/Local |
| `src/api.py` | API REST FastAPI avec validation Pydantic |
| `src/data_preprocessing.py` | Pipeline sklearn : encoding, scaling, feature engineering |
| `src/utils.py` | Constantes, chemins, configuration centralisée |
| `airflow/dags/churn_dag_improved.py` | DAG Airflow complet (9 étapes) |
| `docker/docker-compose-hybrid.yml` | Configuration Docker optimisée Windows |
| `tests/test_*.py` | Tests unitaires et d'intégration |

---

## 📊 Résultats et Métriques

### 🎯 Performance du Modèle

Le modèle **Random Forest** avec les hyperparamètres suivants :
- `n_estimators` : 200
- `max_depth` : 15
- `min_samples_split` : 5
- `random_state` : 42

**Métriques sur le jeu de test (1407 clients) :**

| Métrique | Score | Interprétation |
|----------|-------|----------------|
| **ROC-AUC** | **82.29%** | ⭐⭐⭐⭐ Excellente capacité de discrimination |
| **Accuracy** | **79.03%** | ⭐⭐⭐⭐ Très bonne précision globale |
| **Precision** | **62.95%** | ⭐⭐⭐ Sur 100 prédictions de churn, 63 sont correctes |
| **Recall** | **51.34%** | ⭐⭐⭐ Détecte 51% des vrais churners |
| **F1-Score** | **56.55%** | ⭐⭐⭐ Bon équilibre precision/recall |

### 📈 Matrice de Confusion

```
                 Prédiction
                No Churn | Churn
    ┌─────────┬──────────┬────────┐
 R  │ No      │   959    │   78   │
 é  │ Churn   │          │        │
 e  ├─────────┼──────────┼────────┤
 l  │ Churn   │   181    │  189   │
 i  │         │          │        │
 t  └─────────┴──────────┴────────┘
 é

- Vrais Négatifs (TN) : 959
- Faux Positifs (FP) : 78
- Faux Négatifs (FN) : 181
- Vrais Positifs (TP) : 189
```
---

