import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")


def run_eda(df: pd.DataFrame, results_dir: str = "results") -> None:
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print(f"\n[EDA] Dataset shape    : {df.shape}")
    print(f"[EDA] Missing values   : {df.isnull().sum().sum()}")
    print(f"[EDA] Duplicate rows   : {df.duplicated().sum()}")
    print(f"\n[EDA] Attack type distribution (top 15):\n")
    print(df["label"].value_counts().head(15).to_string())

    binary_labels = df["label"].apply(
        lambda x: "Normal" if str(x).strip().lower() == "normal" else "Attack"
    )
    counts = binary_labels.value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=["#4C9BE8", "#E8604C"], edgecolor="white", linewidth=0.8)
    ax.set_title("Normal vs Attack Traffic Distribution", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Traffic Class", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)

    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val:,} ({val/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10
        )
    plt.tight_layout()
    path = os.path.join(results_dir, "attack_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[EDA] Saved → {path}")

    numeric_df = df.select_dtypes(include="number")

    corr_matrix = numeric_df.corr().abs()
    top_features = corr_matrix.sum().nlargest(20).index.tolist()
    top_corr = numeric_df[top_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        top_corr, annot=False, fmt=".2f", cmap="coolwarm",
        linewidths=0.3, ax=ax, square=True,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap (Top 20 Features)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    path = os.path.join(results_dir, "top_feature_correlation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved → {path}")

    if "protocol_type" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        proto_counts = df.groupby(["protocol_type", binary_labels]).size().unstack(fill_value=0)
        proto_counts.plot(kind="bar", ax=ax, color=["#E8604C", "#4C9BE8"], edgecolor="white")
        ax.set_title("Traffic Class by Protocol Type", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Protocol Type", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(title="Class")
        plt.xticks(rotation=0)
        plt.tight_layout()
        path = os.path.join(results_dir, "protocol_type_dist.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[EDA] Saved → {path}")

    print("\n[EDA] Complete.\n")
