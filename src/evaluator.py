import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


class Evaluator:
    def __init__(
        self,
        models: dict,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        results_dir: str = "results"
    ):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.results_df: pd.DataFrame | None = None
        self._predictions: dict = {}
        self._proba: dict = {}

    def evaluate_all(self) -> pd.DataFrame:
        print("=" * 60)
        print("  MODEL EVALUATION")
        print("=" * 60)

        rows = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            self._predictions[name] = y_pred

            if hasattr(model, "predict_proba"):
                self._proba[name] = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                self._proba[name] = model.decision_function(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)

            rows.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
            })

            print(f"\n  [{name}]")
            print(f"    Accuracy  : {acc:.4f}")
            print(f"    Precision : {prec:.4f}")
            print(f"    Recall    : {rec:.4f}")
            print(f"    F1-Score  : {f1:.4f}")

        self.results_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)

        print("\n[Evaluator] Summary (sorted by F1):")
        print(self.results_df.to_string(index=False))
        print()

        csv_path = os.path.join(self.results_dir, "model_metrics.csv")
        self.results_df.to_csv(csv_path, index=False)
        print(f"[Evaluator] Metrics saved → {csv_path}\n")

        return self.results_df

    def best_model_name(self) -> str:
        if self.results_df is None:
            raise RuntimeError("Call evaluate_all() first.")
        return self.results_df.iloc[0]["Model"]

    def plot_confusion_matrices(self) -> None:
        for name, y_pred in self._predictions.items():
            cm = confusion_matrix(self.y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"],
                linewidths=0.5
            )
            ax.set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel("Predicted Label", fontsize=10)
            ax.set_ylabel("True Label", fontsize=10)
            plt.tight_layout()
            safe_name = name.replace(" ", "_")
            path = os.path.join(self.results_dir, f"confusion_matrix_{safe_name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Evaluator] Saved → {path}")

    def plot_roc_curves(self) -> None:
        colors = ["#4C9BE8", "#E8604C", "#50C878", "#F5A623", "#9B59B6"]
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, (name, proba) in enumerate(self._proba.items()):
            fpr, tpr, _ = roc_curve(self.y_test, proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f"{name}  (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()

        path = os.path.join(self.results_dir, "roc_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Evaluator] Saved → {path}")

    def plot_model_comparison(self) -> None:
        if self.results_df is None:
            raise RuntimeError("Call evaluate_all() first.")

        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        df = self.results_df.set_index("Model")[metrics]

        x = np.arange(len(df))
        width = 0.18
        colors = ["#4C9BE8", "#50C878", "#F5A623", "#E8604C"]

        fig, ax = plt.subplots(figsize=(11, 6))
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, df[metric], width, label=metric,
                          color=color, edgecolor="white", linewidth=0.5)
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90
                )

        ax.set_xticks(x)
        ax.set_xticklabels(df.index, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="lower right", fontsize=9)
        ax.axhline(0.95, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        plt.tight_layout()

        path = os.path.join(self.results_dir, "model_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Evaluator] Saved → {path}")
