"""
API FastAPI pour servir le modèle de prédiction de churn.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import joblib
import pandas as pd
from pathlib import Path
import json

from src.utils import MODEL_PATH, METRICS_PATH, get_project_info

# Initialisation de l'application
app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction du churn client pour les télécommunications",
    version="1.0.0"
)

# Chargement du modèle au démarrage
model = None


@app.on_event("startup")
async def load_model():
    """Charge le modèle au démarrage de l'API."""
    global model
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            print(f"✅ Modèle chargé depuis {MODEL_PATH}")
        else:
            print(f"⚠️ Aucun modèle trouvé à {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {str(e)}")


# Schémas Pydantic pour la validation
class CustomerData(BaseModel):
    """Données d'un client pour la prédiction."""
    gender: str = Field(..., description="Male ou Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 ou 1")
    Partner: str = Field(..., description="Yes ou No")
    Dependents: str = Field(..., description="Yes ou No")
    tenure: int = Field(..., ge=0, description="Nombre de mois d'ancienneté")
    PhoneService: str = Field(..., description="Yes ou No")
    MultipleLines: str = Field(..., description="Yes, No, ou No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, ou No")
    OnlineSecurity: str = Field(..., description="Yes, No, ou No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, ou No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, ou No internet service")
    TechSupport: str = Field(..., description="Yes, No, ou No internet service")
    StreamingTV: str = Field(..., description="Yes, No, ou No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, ou No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, ou Two year")
    PaperlessBilling: str = Field(..., description="Yes ou No")
    PaymentMethod: str = Field(..., description="Méthode de paiement")
    MonthlyCharges: float = Field(..., ge=0, description="Montant mensuel")
    TotalCharges: float = Field(..., ge=0, description="Montant total")

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    churn_probability: float = Field(..., description="Probabilité de churn (0-1)")
    churn_prediction: str = Field(..., description="Yes ou No")
    risk_level: str = Field(..., description="Low, Medium, ou High")


class BatchPredictionRequest(BaseModel):
    """Requête pour prédictions batch."""
    customers: List[CustomerData]


# Routes de l'API
@app.get("/")
async def root():
    """Route racine de l'API."""
    return {
        "message": "API de prédiction de churn - Télécommunications",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "metrics": "/metrics",
            "info": "/info"
        }
    }


@app.get("/health")
async def health_check():
    """Vérification de santé de l'API."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


@app.get("/info")
async def get_info():
    """Informations sur le projet et le modèle."""
    info = get_project_info()
    
    # Charger les métriques si disponibles
    if METRICS_PATH.exists():
        with open(METRICS_PATH, 'r') as f:
            info["metrics"] = json.load(f)
    
    return info


@app.get("/metrics")
async def get_metrics():
    """Récupère les métriques du modèle."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Métriques non trouvées")
    
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    
    return metrics


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """
    Prédit le churn pour un client donné.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Conversion en DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Prédiction
        proba = model.predict_proba(df)[0, 1]
        prediction = "Yes" if proba >= 0.5 else "No"
        
        # Détermination du niveau de risque
        if proba < 0.3:
            risk_level = "Low"
        elif proba < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            churn_probability=float(proba),
            churn_prediction=prediction,
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """
    Prédit le churn pour plusieurs clients.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Conversion en DataFrame
        customers_data = [customer.dict() for customer in request.customers]
        df = pd.DataFrame(customers_data)
        
        # Prédictions
        probas = model.predict_proba(df)[:, 1]
        predictions = ["Yes" if p >= 0.5 else "No" for p in probas]
        
        # Niveaux de risque
        risk_levels = []
        for proba in probas:
            if proba < 0.3:
                risk_levels.append("Low")
            elif proba < 0.7:
                risk_levels.append("Medium")
            else:
                risk_levels.append("High")
        
        results = [
            {
                "churn_probability": float(prob),
                "churn_prediction": pred,
                "risk_level": risk
            }
            for prob, pred, risk in zip(probas, predictions, risk_levels)
        ]
        
        return {
            "predictions": results,
            "summary": {
                "total_customers": len(results),
                "predicted_churners": sum(1 for p in predictions if p == "Yes"),
                "churn_rate": sum(1 for p in predictions if p == "Yes") / len(predictions)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from src.utils import API_HOST, API_PORT
    
    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )