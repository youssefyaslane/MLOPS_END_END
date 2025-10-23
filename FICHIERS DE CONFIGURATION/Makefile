.PHONY: help install test train api docker-up docker-down clean lint format

help:
	@echo "📋 Commandes disponibles:"
	@echo "  make install      - Installer les dépendances"
	@echo "  make test         - Exécuter les tests"
	@echo "  make test-cov     - Exécuter les tests avec couverture"
	@echo "  make train        - Entraîner le modèle"
	@echo "  make api          - Lancer l'API FastAPI"
	@echo "  make docker-up    - Démarrer tous les services Docker"
	@echo "  make docker-down  - Arrêter tous les services Docker"
	@echo "  make mlflow       - Lancer MLflow UI"
	@echo "  make airflow      - Accéder à Airflow Web UI"
	@echo "  make clean        - Nettoyer les fichiers générés"
	@echo "  make lint         - Vérifier le code avec flake8"
	@echo "  make format       - Formatter le code avec black"

install:
	@echo "📦 Installation des dépendances..."
	pip install -r requirements.txt
	@echo "✅ Dépendances installées"

test:
	@echo "🧪 Exécution des tests..."
	pytest tests/ -v

test-cov:
	@echo "🧪 Exécution des tests avec couverture..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "📊 Rapport de couverture généré dans htmlcov/index.html"

train:
	@echo "🎯 Entraînement du modèle..."
	python -m src.train

api:
	@echo "🚀 Lancement de l'API..."
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	@echo "🐳 Démarrage des services Docker..."
	cd docker && docker-compose -f docker-compose-complete.yml up -d
	@echo "✅ Services démarrés:"
	@echo "  - Airflow Web UI: http://localhost:8080 (admin/admin)"
	@echo "  - MLflow UI: http://localhost:5000"
	@echo "  - API: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"

docker-down:
	@echo "🐳 Arrêt des services Docker..."
	cd docker && docker-compose -f docker-compose-complete.yml down
	@echo "✅ Services arrêtés"

docker-clean:
	@echo "🧹 Nettoyage complet Docker..."
	cd docker && docker-compose -f docker-compose-complete.yml down -v
	@echo "✅ Volumes supprimés"

mlflow:
	@echo "🔬 MLflow UI disponible sur: http://localhost:5000"
	@xdg-open http://localhost:5000 2>/dev/null || open http://localhost:5000 2>/dev/null || echo "Ouvrez http://localhost:5000 dans votre navigateur"

airflow:
	@echo "🌊 Airflow Web UI disponible sur: http://localhost:8080"
	@echo "   Identifiants: admin/admin"
	@xdg-open http://localhost:8080 2>/dev/null || open http://localhost:8080 2>/dev/null || echo "Ouvrez http://localhost:8080 dans votre navigateur"

api-docs:
	@echo "📚 Documentation API disponible sur: http://localhost:8000/docs"
	@xdg-open http://localhost:8000/docs 2>/dev/null || open http://localhost:8000/docs 2>/dev/null || echo "Ouvrez http://localhost:8000/docs dans votre navigateur"

clean:
	@echo "🧹 Nettoyage des fichiers générés..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache/
	@echo "✅ Nettoyage terminé"

lint:
	@echo "🔍 Vérification du code..."
	flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503

format:
	@echo "✨ Formatage du code..."
	black src/ tests/ --line-length=120
	@echo "✅ Code formaté"

setup-env:
	@echo "⚙️ Configuration de l'environnement..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Fichier .env créé. Veuillez le configurer."; \
	else \
		echo "⚠️ Fichier .env existe déjà."; \
	fi

logs-airflow:
	@echo "📋 Logs Airflow..."
	cd docker && docker-compose -f docker-compose-complete.yml logs -f airflow-scheduler

logs-api:
	@echo "📋 Logs API..."
	cd docker && docker-compose -f docker-compose-complete.yml logs -f api

logs-mlflow:
	@echo "📋 Logs MLflow..."
	cd docker && docker-compose -f docker-compose-complete.yml logs -f mlflow

all: install test train
	@echo "✅ Installation, tests et entraînement terminés"