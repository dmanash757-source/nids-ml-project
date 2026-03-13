import joblib
import pandas as pd
import numpy as np


class IntrusionPredictor:
    def __init__(self, model_path: str, preprocessor, selected_features: list = None):
        self.model = joblib.load(model_path)
        self.preprocessor = preprocessor
        self.selected_features = selected_features
        print(f"[Predictor] Model loaded from '{model_path}'")
        print(f"[Predictor] Model type: {type(self.model).__name__}\n")

    def predict_one(self, sample: dict) -> dict:
        df = pd.DataFrame([sample])

        df.drop(columns=["label"], inplace=True, errors="ignore")

        X, _ = self.preprocessor.transform(df)

        if self.selected_features:
            X = X[self.selected_features]

        y_pred = self.model.predict(X)[0]
        label = "Normal" if y_pred == 0 else "Attack"

        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[y_pred])

        return {"prediction": label, "confidence": confidence}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df_work = df.copy()
        df_work.drop(columns=["label"], inplace=True, errors="ignore")

        X, _ = self.preprocessor.transform(df_work)

        preds = self.model.predict(X)
        labels = ["Normal" if p == 0 else "Attack" for p in preds]

        result = df.copy()
        result["prediction"] = labels

        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            result["confidence"] = [probas[i, preds[i]] for i in range(len(preds))]

        return result


DEMO_NORMAL_SAMPLE = {
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
    "src_bytes": 215, "dst_bytes": 45076, "land": 0, "wrong_fragment": 0,
    "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
    "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
    "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
    "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 2, "srv_count": 2, "serror_rate": 0.0, "srv_serror_rate": 0.0,
    "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 1,
    "dst_host_srv_count": 1, "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.0,
    "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}

DEMO_ATTACK_SAMPLE = {
    "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "S0",
    "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
    "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
    "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
    "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
    "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 123, "srv_count": 6, "serror_rate": 1.0, "srv_serror_rate": 1.0,
    "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 0.05,
    "diff_srv_rate": 0.07, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
    "dst_host_srv_count": 6, "dst_host_same_srv_rate": 0.02,
    "dst_host_diff_srv_rate": 0.07, "dst_host_same_src_port_rate": 0.0,
    "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0,
    "dst_host_srv_serror_rate": 1.0, "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}
