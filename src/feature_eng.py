import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


class FeatureEngineer:
    def __init__(self, top_n: int = 20, random_state: int = 42):
        self.top_n = top_n
        self.random_state = random_state
        self.selected_features: list[str] = []
        self.importances: pd.Series | None = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=self.random_state
        )
        rf.fit(X, y)

        self.importances = pd.Series(
            rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        self.selected_features = self.importances.head(self.top_n).index.tolist()

        print(f"[FeatureEng] Total features    : {X.shape[1]}")
        print(f"[FeatureEng] Selected features : {len(self.selected_features)}")
        print(f"[FeatureEng] Top 10 features   :")
        for rank, (feat, imp) in enumerate(self.importances.head(10).items(), 1):
            print(f"             {rank:>2}. {feat:<35} {imp:.4f}")
        print()

        return X[self.selected_features]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features:
            raise RuntimeError("Call fit_transform() before transform().")
        return X[self.selected_features]

    def plot_importance(self, results_dir: str = "results") -> None:
        if self.importances is None:
            raise RuntimeError("No importances computed yet.")

        os.makedirs(results_dir, exist_ok=True)

        top = self.importances.head(self.top_n)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))[::-1]

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1],
                edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Feature Importance Score", fontsize=11)
        ax.set_title(f"Top {self.top_n} Feature Importances (Random Forest)",
                     fontsize=13, fontweight="bold", pad=12)

        for i, (val, label) in enumerate(zip(top.values[::-1], top.index[::-1])):
            ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=8)

        plt.tight_layout()
        path = os.path.join(results_dir, "feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[FeatureEng] Saved → {path}")
