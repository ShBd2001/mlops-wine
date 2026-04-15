"""
Script d'entraînement :
- charge le dataset Wine (sklearn)
- applique un prétraitement (StandardScaler)
- entraîne un RandomForestClassifier
- évalue et affiche accuracy + F1-score
- sauvegarde le modèle et le scaler dans artifacts/
"""

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
import json
import os


def main():
    # 1. Chargement du dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Pipeline : StandardScaler + RandomForest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 4. Entraînement
    pipeline.fit(X_train, y_train)

    # 5. Évaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy  : {acc:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # 6. Sauvegarde des artefacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, "artifacts/model.pkl")

    metrics = {"accuracy": round(float(acc), 4), "f1_score": round(float(f1), 4)}
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Modèle sauvegardé dans artifacts/model.pkl")
    print("Métriques sauvegardées dans artifacts/metrics.json")


if __name__ == "__main__":
    main()
