import os
import sys
import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data
from src.preprocessor import Preprocessor
from src.eda import run_eda
from src.feature_eng import FeatureEngineer
from src.trainer import train_all_models
from src.evaluator import Evaluator
from src.predictor import IntrusionPredictor, DEMO_NORMAL_SAMPLE, DEMO_ATTACK_SAMPLE


TRAIN_PATH = os.path.join(PROJECT_ROOT, "dataset", "train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "dataset", "test.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "intrusion_model.pkl")
PREP_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def main():
    print("\n" + "=" * 60)
    print("  ML-BASED NETWORK INTRUSION DETECTION SYSTEM (NIDS)")
    print("  NSL-KDD Dataset | Binary Classification")
    print("=" * 60 + "\n")

    print("─" * 40)
    print("STEP 1 — Loading dataset")
    print("─" * 40)
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    print("─" * 40)
    print("STEP 2 — Exploratory Data Analysis")
    print("─" * 40)
    run_eda(train_df, results_dir=RESULTS_DIR)

    print("─" * 40)
    print("STEP 3 — Preprocessing")
    print("─" * 40)
    prep = Preprocessor()
    X_train, y_train = prep.fit_transform(train_df)
    X_test, y_test = prep.transform(test_df)

    os.makedirs(os.path.dirname(PREP_PATH), exist_ok=True)
    joblib.dump(prep, PREP_PATH)
    print(f"[Main] Preprocessor saved → {PREP_PATH}\n")

    print("─" * 40)
    print("STEP 4 — Feature Engineering")
    print("─" * 40)
    fe = FeatureEngineer(top_n=20)
    X_train_sel = fe.fit_transform(X_train, y_train)
    X_test_sel = fe.transform(X_test)
    fe.plot_importance(results_dir=RESULTS_DIR)

    print("─" * 40)
    print("STEP 5 — Training Models")
    print("─" * 40)
    models = train_all_models(X_train_sel, y_train)

    print("─" * 40)
    print("STEP 6 — Evaluation")
    print("─" * 40)
    evaluator = Evaluator(models, X_test_sel, y_test, results_dir=RESULTS_DIR)
    results_df = evaluator.evaluate_all()
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_model_comparison()

    print("─" * 40)
    print("STEP 7 — Saving Best Model")
    print("─" * 40)
    best_name = evaluator.best_model_name()
    best_model = models[best_name]

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"[Main] Best model  : {best_name}")
    best_row = results_df[results_df["Model"] == best_name].iloc[0]
    print(f"[Main] F1-Score    : {best_row['F1-Score']:.4f}")
    print(f"[Main] Accuracy    : {best_row['Accuracy']:.4f}")
    print(f"[Main] Model saved → {MODEL_PATH}\n")

    print("─" * 40)
    print("STEP 8 — Demo Prediction")
    print("─" * 40)
    predictor = IntrusionPredictor(model_path=MODEL_PATH, preprocessor=prep, selected_features=fe.selected_features)

    for label, sample in [("NORMAL sample", DEMO_NORMAL_SAMPLE),
                           ("ATTACK sample", DEMO_ATTACK_SAMPLE)]:
        result = predictor.predict_one(sample)
        conf = f"{result['confidence']:.2%}" if result["confidence"] is not None else "N/A"
        print(f"  {label:<20} → Prediction: {result['prediction']:<8}  Confidence: {conf}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Plots   → {RESULTS_DIR}/")
    print(f"  Model   → {MODEL_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
