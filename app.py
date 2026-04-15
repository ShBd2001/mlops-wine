"""
API FastAPI pour le service de prédiction Wine.

Endpoints :
  GET  /health   → statut du service
  POST /predict  → prédiction à partir des features
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# App & modèle
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Wine Prediction API",
    description="Prédit la classe d'un vin (0, 1 ou 2) à partir de 13 features.",
    version="1.0.0",
)

MODEL_PATH = Path("artifacts/model.pkl")

# Chargement du modèle au démarrage
if not MODEL_PATH.exists():
    # Si le modèle n'existe pas, on l'entraîne automatiquement
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH)

# Noms des classes Wine
CLASS_NAMES = ["class_0", "class_1", "class_2"]

# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=13,
        max_length=13,
        description="13 features du dataset Wine dans l'ordre standard",
        example=[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820.0]
    )

class PredictResponse(BaseModel):
    prediction: int
    label: str
    probabilities: List[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Vérification de l'état du service")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictResponse, summary="Prédiction de classe")
def predict(body: PredictRequest):
    try:
        X = np.array(body.features).reshape(1, -1)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0].tolist()
        return PredictResponse(
            prediction=int(pred),
            label=CLASS_NAMES[int(pred)],
            probabilities=[round(p, 4) for p in proba],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
