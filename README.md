# Machine Learning Based Network Intrusion Detection System (NIDS)

A complete machine learning pipeline for detecting network intrusions using the NSL-KDD dataset. This project implements binary classification to distinguish between normal network traffic and attack traffic.

## Overview

This project demonstrates a complete ML pipeline:
- **Data Loading**: NSL-KDD dataset with 41 features
- **Preprocessing**: Categorical encoding, normalization, binary label conversion
- **Feature Engineering**: Random Forest-based feature selection (top 20 features)
- **Model Training**: Decision Tree, Random Forest, Logistic Regression, KNN
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC curves, Confusion Matrices
- **Prediction**: Real-time intrusion detection on new network traffic samples

## Project Structure

```
NIDS_Project/
├── dataset/               # NSL-KDD train.csv and test.csv
├── models/               # Trained models (intrusion_model.pkl, preprocessor.pkl)
├── results/              # Evaluation plots and metrics
├── src/
│   ├── data_loader.py    # Dataset loading
│   ├── preprocessor.py   # Data preprocessing
│   ├── eda.py            # Exploratory Data Analysis
│   ├── feature_eng.py    # Feature selection
│   ├── trainer.py        # Model training
│   ├── evaluator.py      # Model evaluation
│   └── predictor.py     # Inference module
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── .gitignore
```

## Dataset

**NSL-KDD** is an improved version of the KDD Cup 1999 dataset for network intrusion detection.

- **Training samples**: 125,973
- **Test samples**: 22,544
- **Features**: 41 (numerical + categorical)
- **Classes**: Normal + 22 attack types (converted to binary: Normal/Attack)

## Models Trained

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 78.81% | 91.79% | 68.95% | 78.75% |
| KNN | 78.99% | 96.94% | 65.14% | 77.92% |
| Logistic Regression | 75.11% | 92.43% | 61.30% | 73.71% |
| Random Forest | 75.53% | 96.22% | 59.35% | 73.41% |

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Ensure dataset files are in dataset/ folder
# - dataset/train.csv
# - dataset/test.csv
```

## Usage

### Run Full Pipeline

```bash
python main.py
```

This executes:
1. Data loading and exploration
2. Preprocessing (encoding, normalization)
3. Feature selection (top 20 features)
4. Model training (4 classifiers)
5. Evaluation (metrics, plots)
6. Model saving
7. Demo predictions

### Programmatic Usage

```python
from src.predictor import IntrusionPredictor
from src.preprocessor import Preprocessor
import joblib

# Load saved model and preprocessor
model = joblib.load("models/intrusion_model.pkl")
prep = joblib.load("models/preprocessor.pkl")

# Create predictor
predictor = IntrusionPredictor(
    model_path="models/intrusion_model.pkl",
    preprocessor=prep,
    selected_features=['flag', 'same_srv_rate', ...]  # 20 selected features
)

# Predict on new traffic sample
sample = {
    "duration": 0, "protocol_type": "tcp", "service": "http",
    "flag": "SF", "src_bytes": 215, "dst_bytes": 45076, ...
}
result = predictor.predict_one(sample)
print(result)  # {'prediction': 'Normal', 'confidence': 0.95}
```

## Results

All evaluation results are saved in `results/`:
- `attack_distribution.png` - Class distribution visualization
- `top_feature_correlation.png` - Feature correlation heatmap
- `protocol_type_dist.png` - Traffic type breakdown
- `feature_importance.png` - Top 20 selected features
- `confusion_matrix_*.png` - Confusion matrices per model
- `roc_curves.png` - ROC curves comparison
- `model_comparison.png` - Model performance comparison
- `model_metrics.csv` - Raw metrics table

## Requirements

- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- joblib >= 1.2.0

## Author

- **Project**: Master's Final Year Project
- **Domain**: Network Security / Machine Learning

## License

This project is for educational purposes.
