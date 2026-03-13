import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


class Preprocessor:
    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {
            col: LabelEncoder() for col in CATEGORICAL_COLS
        }
        self.scaler = MinMaxScaler()
        self.feature_columns: list[str] = []

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if fit:
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                encoder = self.label_encoders[col]
                known = set(encoder.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else encoder.classes_[0]
                )
                df[col] = encoder.transform(df[col])
        return df

    @staticmethod
    def _binarise_labels(series: pd.Series) -> pd.Series:
        return series.apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)

    def _scale_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            scaled = self.scaler.fit_transform(X)
        else:
            scaled = self.scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()

        y = self._binarise_labels(df["label"])
        df.drop(columns=["label"], inplace=True)

        df = self._encode_categoricals(df, fit=True)

        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        self.feature_columns = df.columns.tolist()
        X = self._scale_features(df, fit=True)

        print(f"[Preprocessor] Features : {X.shape[1]}")
        print(f"[Preprocessor] Samples  : {X.shape[0]}")
        print(f"[Preprocessor] Class balance → Normal: {(y==0).sum()}, Attack: {(y==1).sum()}\n")
        return X, y

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()

        if "label" in df.columns:
            y = self._binarise_labels(df["label"])
            df.drop(columns=["label"], inplace=True)
        else:
            y = pd.Series(dtype=int)

        df = self._encode_categoricals(df, fit=False)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]

        X = self._scale_features(df, fit=False)
        return X, y
