import json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from src.data_preprocessing import load_data, make_preprocessor, split_data
from src.utils import MODEL_PATH, METRICS_PATH, DATA_PATH

def main():
    df = load_data(str(DATA_PATH))
    X, y, pre = make_preprocessor(df)
    X_train, X_test, y_train, y_test = split_data(X, y)test_data.py

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    pipe = Pipeline([("prep", pre), ("clf", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {"roc_auc": float(roc_auc_score(y_test, proba))}

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, MODEL_PATH)
    print(f"✅ {MODEL_PATH} | AUC={metrics['roc_auc']:.3f}")

if __name__ == "__main__":
    main()
