#!/bin/bash

# Script de démarrage rapide pour le projet MLOps Churn
set -e

echo "🚀 ============================================="
echo "   MLOPS CHURN TELECOM - Démarrage rapide"
echo "============================================="
echo ""

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher les étapes
step() {
    echo -e "${BLUE}▶ $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 1. Vérifier les prérequis
step "Vérification des prérequis..."

if ! command -v python3 &> /dev/null; then
    warning "Python 3 n'est pas installé!"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    warning "Docker n'est pas installé!"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    warning "Docker Compose n'est pas installé!"
    exit 1
fi

success "Tous les prérequis sont installés"
echo ""

# 2. Créer l'environnement virtuel
step "Création de l'environnement virtuel..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    success "Environnement virtuel créé"
else
    success "Environnement virtuel déjà existant"
fi
echo ""

# 3. Activer l'environnement virtuel
step "Activation de l'environnement virtuel..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null || {
    warning "Impossible d'activer l'environnement virtuel"
    echo "Veuillez l'activer manuellement:"
    echo "  Linux/Mac: source venv/bin/activate"
    echo "  Windows: venv\\Scripts\\activate"
}
echo ""

# 4. Installer les dépendances
step "Installation des dépendances Python..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
success "Dépendances installées"
echo ""

# 5. Créer le fichier .env si nécessaire
step "Configuration de l'environnement..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    success "Fichier .env créé"
    warning "Pensez à configurer vos variables d'environnement dans .env"
else
    success "Fichier .env déjà existant"
fi
echo ""

# 6. Créer les répertoires nécessaires
step "Création des répertoires..."
mkdir -p data artifacts/models artifacts/archive airflow/dags tests
success "Répertoires créés"
echo ""

# 7. Vérifier la présence des données
step "Vérification des données..."
if [ ! -f "data/telco_churn.csv" ]; then
    warning "Le fichier de données data/telco_churn.csv n'existe pas"
    echo "Veuillez télécharger le dataset et le placer dans data/"
else
    success "Dataset trouvé"
fi
echo ""

# 8. Démarrer Docker (optionnel)
echo "Voulez-vous démarrer les services Docker ? (y/n)"
read -p "Réponse: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    step "Démarrage des services Docker..."
    cd docker
    docker-compose -f docker-compose-complete.yml up -d
    cd ..
    success "Services Docker démarrés"
    echo ""
    echo "📊 Services disponibles:"
    echo "  - Airflow Web UI: http://localhost:8080 (admin/admin)"
    echo "  - MLflow UI: http://localhost:5000"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
fi

echo ""
echo "🎉 ============================================="
echo "   Configuration terminée avec succès!"
echo "============================================="
echo ""
echo "📚 Prochaines étapes:"
echo ""
echo "1. Activer l'environnement virtuel (si pas déjà fait):"
echo "   source venv/bin/activate"
echo ""
echo "2. Entraîner le modèle:"
echo "   make train"
echo "   # ou"
echo "   python -m src.train"
echo ""
echo "3. Lancer l'API:"
echo "   make api"
echo "   # ou"
echo "   uvicorn src.api:app --reload"
echo ""
echo "4. Exécuter les tests:"
echo "   make test"
echo ""
echo "5. Voir toutes les commandes disponibles:"
echo "   make help"
echo ""
echo "📖 Consultez le README.md pour plus d'informations"
echo ""