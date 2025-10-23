"""
Tests pour l'API FastAPI de prédiction de churn.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app

client = TestClient(app)


class TestAPIEndpoints:
    """Tests des endpoints de l'API."""
    
    def test_root_endpoint(self):
        """Test de l'endpoint racine."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "running"
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test de l'endpoint de santé."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_info_endpoint(self):
        """Test de l'endpoint d'informations."""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "project_name" in data
        assert "environment" in data
    
    def test_metrics_endpoint_when_available(self):
        """Test de l'endpoint de métriques (si disponible)."""
        response = client.get("/metrics")
        
        # Peut retourner 200 ou 404 selon si le modèle a été entraîné
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)


class TestPredictionEndpoint:
    """Tests de l'endpoint de prédiction."""
    
    @pytest.fixture
    def valid_customer_data(self):
        """Données valides d'un client."""
        return {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": 844.2
        }
    
    def test_predict_valid_data(self, valid_customer_data):
        """Test de prédiction avec données valides."""
        response = client.post("/predict", json=valid_customer_data)
        
        # Peut retourner 200 ou 503 selon si le modèle est chargé
        if response.status_code == 200:
            data = response.json()
            assert "churn_probability" in data
            assert "churn_prediction" in data
            assert "risk_level" in data
            
            assert 0 <= data["churn_probability"] <= 1
            assert data["churn_prediction"] in ["Yes", "No"]
            assert data["risk_level"] in ["Low", "Medium", "High"]
        else:
            assert response.status_code == 503
            assert "Modèle non chargé" in response.json()["detail"]
    
    def test_predict_invalid_data_missing_field(self):
        """Test avec données invalides (champ manquant)."""
        invalid_data = {
            "gender": "Female",
            "SeniorCitizen": 0,
            # Champs manquants...
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_data_wrong_type(self):
        """Test avec type de données incorrect."""
        invalid_data = {
            "gender": "Female",
            "SeniorCitizen": "zero",  # Devrait être int
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35,
            "TotalCharges": 844.2
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422


class TestBatchPrediction:
    """Tests de l'endpoint de prédiction batch."""
    
    @pytest.fixture
    def valid_batch_data(self):
        """Données batch valides."""
        return {
            "customers": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.35,
                    "TotalCharges": 844.2
                },
                {
                    "gender": "Male",
                    "SeniorCitizen": 1,
                    "Partner": "No",
                    "Dependents": "No",
                    "tenure": 48,
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Two year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Bank transfer",
                    "MonthlyCharges": 85.0,
                    "TotalCharges": 4080.0
                }
            ]
        }
    
    def test_batch_predict_valid_data(self, valid_batch_data):
        """Test de prédiction batch avec données valides."""
        response = client.post("/predict/batch", json=valid_batch_data)
        
        # Peut retourner 200 ou 503 selon si le modèle est chargé
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "summary" in data
            
            assert len(data["predictions"]) == 2
            
            summary = data["summary"]
            assert summary["total_customers"] == 2
            assert "predicted_churners" in summary
            assert "churn_rate" in summary
        else:
            assert response.status_code == 503
    
    def test_batch_predict_empty_list(self):
        """Test avec liste vide."""
        response = client.post("/predict/batch", json={"customers": []})
        
        # Peut retourner 200 avec résultats vides ou 422
        assert response.status_code in [200, 422, 503]


class TestAPIDocumentation:
    """Tests de la documentation de l'API."""
    
    def test_openapi_schema(self):
        """Test de la disponibilité du schéma OpenAPI."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_docs_endpoint(self):
        """Test de la page de documentation."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint(self):
        """Test de la page ReDoc."""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.api", "--cov-report=html"])