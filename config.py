import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_PATH = os.path.join(DATASET_DIR, "train.csv")
TEST_PATH = os.path.join(DATASET_DIR, "test.csv")

# Model paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "intrusion_model.pkl")
PREP_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

# Results path
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Feature engineering
TOP_N_FEATURES = 20
RANDOM_STATE = 42

# Model hyperparameters
MODEL_CONFIG = {
    "decision_tree": {
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
    },
    "logistic_regression": {
        "max_iter": 1000,
        "solver": "lbfgs",
        "C": 1.0,
    },
    "knn": {
        "n_neighbors": 5,
        "metric": "euclidean",
    }
}
