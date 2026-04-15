# MLOps Wine — Mini projet MLOps

Service de prédiction de classe de vin basé sur le dataset **Wine** de scikit-learn.

## Stack
- **Python 3.11** + scikit-learn + FastAPI
- **Docker**
- **GitHub Actions** (CI)

## Dataset
Le dataset Wine contient 178 échantillons de vins italiens décrits par 13 features chimiques.  
Le modèle (Random Forest + StandardScaler) prédit l'une des 3 classes de vin.

## Lancer en local

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Entraîner le modèle
python train.py

# 3. Démarrer l'API
uvicorn app:app --reload --port 8000
```

## Endpoints

| Méthode | Route      | Description            |
|---------|------------|------------------------|
| GET     | `/health`  | Santé du service       |
| POST    | `/predict` | Prédiction de classe   |

### Exemple de requête `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820.0]}'
```

### Réponse

```json
{
  "prediction": 0,
  "label": "class_0",
  "probabilities": [0.97, 0.02, 0.01]
}
```

## Docker

```bash
docker build -t mlops-wine .
docker run -p 8000:8000 mlops-wine
```

## CI/CD (GitHub Actions)

| Branche       | Jobs déclenchés                                |
|---------------|------------------------------------------------|
| `feature/**`  | install deps → train model                     |
| `develop`     | install deps → train model → build → push image|

### Secrets à configurer dans GitHub

- `DOCKERHUB_USERNAME` : ton nom d'utilisateur Docker Hub
- `DOCKERHUB_TOKEN` : ton access token Docker Hub (Settings → Security)
