import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


MODEL_CATALOGUE = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",
        n_jobs=-1
    ),
}


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> dict:
    fitted_models = {}

    print("=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)

    for name, model in MODEL_CATALOGUE.items():
        print(f"\n[Trainer] Training: {name} ...", end=" ", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"done  ({elapsed:.1f}s)")
        fitted_models[name] = model

    print("\n[Trainer] All models trained successfully.\n")
    return fitted_models
