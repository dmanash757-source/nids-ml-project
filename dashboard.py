import os
import sys
import json

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

TRAIN_PATH = os.path.join(PROJECT_ROOT, "dataset", "train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "dataset", "test.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "intrusion_model.pkl")
PREP_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")
METRICS_CSV = os.path.join(PROJECT_ROOT, "results", "model_metrics.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

:root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --border:   #1f2d45;
    --accent:   #00d4ff;
    --danger:   #ff4b6e;
    --success:  #00e676;
    --warning:  #ffb300;
    --text:     #c9d6e8;
    --muted:    #5a7090;
    --mono:     'Share Tech Mono', monospace;
    --sans:     'Rajdhani', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--accent) !important;
    border-radius: 6px !important;
    padding: 14px !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--mono) !important; font-size: 26px !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    border-radius: 4px !important;
    letter-spacing: 0.05em;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #0a0e1a !important;
}

.stSelectbox > div, .stNumberInput > div, .stSlider > div {
    background: var(--surface) !important;
}
input, select, textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    font-family: var(--mono) !important;
}

[data-testid="stTabs"] button {
    font-family: var(--mono) !important;
    font-size: 13px !important;
    color: var(--muted) !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

h1, h2, h3 { font-family: var(--sans) !important; color: #e8f0fe !important; font-weight: 700 !important; letter-spacing: 0.04em; }
h1 { font-size: 28px !important; }
h2 { font-size: 20px !important; }

.nids-divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

.pred-attack  { background: rgba(255,75,110,0.15); border: 1px solid var(--danger); border-radius: 6px; padding: 20px; text-align: center; color: var(--danger); font-family: var(--mono); font-size: 22px; font-weight: bold; }
.pred-normal  { background: rgba(0,230,118,0.12); border: 1px solid var(--success); border-radius: 6px; padding: 20px; text-align: center; color: var(--success); font-family: var(--mono); font-size: 22px; font-weight: bold; }
.pred-unknown { background: rgba(90,112,144,0.15); border: 1px solid var(--muted); border-radius: 6px; padding: 20px; text-align: center; color: var(--muted); font-family: var(--mono); font-size: 18px; }

.info-box { background: rgba(0,212,255,0.07); border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0; padding: 12px 16px; font-size: 14px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_raw_data():
    from src.data_loader import load_data
    return load_data(TRAIN_PATH, TEST_PATH)

@st.cache_resource
def load_preprocessor():
    import joblib
    return joblib.load(PREP_PATH)

@st.cache_resource
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_CSV)

with st.sidebar:
    st.markdown("## 🛡️ NIDS Dashboard")
    st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Overview", "EDA & Analysis", "Model Results", "Confusion Matrices", "Live Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)

    st.markdown("**Pipeline Status**")
    checks = {
        "Dataset (train)": os.path.exists(TRAIN_PATH),
        "Dataset (test)": os.path.exists(TEST_PATH),
        "Trained model": os.path.exists(MODEL_PATH),
        "Preprocessor": os.path.exists(PREP_PATH),
        "Metrics CSV": os.path.exists(METRICS_CSV),
    }
    for label, ok in checks.items():
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} `{label}`")

    st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)
    if not all(checks.values()):
        st.warning("Run `python main.py` first to generate models and results.")

if page == "Overview":
    st.markdown("# Network Intrusion Detection System")
    st.markdown("<div class='info-box'>Binary classification on the <b>NSL-KDD</b> dataset — distinguishing <b>Normal</b> vs <b>Attack</b> network traffic using four ML models.</div>", unsafe_allow_html=True)

    if not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)):
        st.error("Dataset not found. Place `train.csv` and `test.csv` in the `dataset/` folder.")
        st.stop()

    train_df, test_df = load_raw_data()

    binary_train = train_df["label"].apply(lambda x: "Normal" if str(x).lower() == "normal" else "Attack")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Train samples", f"{len(train_df):,}")
    c2.metric("Test samples", f"{len(test_df):,}")
    c3.metric("Features", "41")
    c4.metric("Normal (train)", f"{(binary_train=='Normal').sum():,}")
    c5.metric("Attack (train)", f"{(binary_train=='Attack').sum():,}")

    st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Attack Type Distribution")
        top_attacks = train_df["label"].value_counts().head(12)
        colors = ["#00d4ff" if l == "normal" else "#ff4b6e" for l in top_attacks.index]
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#111827")
        ax.set_facecolor("#111827")
        bars = ax.barh(top_attacks.index[::-1], top_attacks.values[::-1], color=colors[::-1], edgecolor="#1f2d45")
        ax.tick_params(colors="#c9d6e8", labelsize=9)
        ax.xaxis.label.set_color("#5a7090")
        for spine in ax.spines.values(): spine.set_edgecolor("#1f2d45")
        ax.set_xlabel("Count", color="#5a7090", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.markdown("#### Feature Categories")
        categories = {
            "Basic features": 9,
            "Content features": 13,
            "Traffic features": 9,
            "Host-based features": 10,
        }
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#111827")
        ax.set_facecolor("#111827")
        wedges, texts, autotexts = ax.pie(
            categories.values(),
            labels=categories.keys(),
            autopct="%1.0f%%",
            colors=["#00d4ff", "#00e676", "#ffb300", "#ff4b6e"],
            startangle=140,
            textprops={"color": "#c9d6e8", "fontsize": 9},
            wedgeprops={"linewidth": 1.5, "edgecolor": "#0a0e1a"}
        )
        for at in autotexts: at.set_color("#0a0e1a")
        ax.set_title("41 Features by Category", color="#c9d6e8", fontsize=10, pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)
    st.markdown("#### Sample Records")
    st.dataframe(train_df.head(8), use_container_width=True)

elif page == "EDA & Analysis":
    st.markdown("# Exploratory Data Analysis")

    plots = {
        "Class Distribution": os.path.join(RESULTS_DIR, "attack_distribution.png"),
        "Feature Correlation Heatmap": os.path.join(RESULTS_DIR, "top_feature_correlation.png"),
        "Protocol Type Breakdown": os.path.join(RESULTS_DIR, "protocol_type_dist.png"),
        "Feature Importances": os.path.join(RESULTS_DIR, "feature_importance.png"),
    }

    missing = [k for k, v in plots.items() if not os.path.exists(v)]
    if missing:
        st.warning(f"Run `python main.py` to generate plots. Missing: {', '.join(missing)}")

    for title, path in plots.items():
        if os.path.exists(path):
            st.markdown(f"#### {title}")
            st.image(path, use_container_width=True)
            st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)

elif page == "Model Results":
    st.markdown("# Model Performance")

    if not os.path.exists(METRICS_CSV):
        st.error("No metrics found. Run `python main.py` first.")
        st.stop()

    df = load_metrics()

    best_idx = df["F1-Score"].idxmax()
    best_row = df.loc[best_idx]

    st.markdown(f"<div class='info-box'>🏆 Best model: <b>{best_row['Model']}</b> — F1 = <b>{best_row['F1-Score']:.4f}</b> | Accuracy = <b>{best_row['Accuracy']:.4f}</b></div>", unsafe_allow_html=True)

    def highlight_best(row):
        style = [""] * len(row)
        if row["Model"] == best_row["Model"]:
            style = ["background-color: rgba(0,212,255,0.08); color: #00d4ff; font-weight: bold"] * len(row)
        return style

    st.dataframe(
        df.style.apply(highlight_best, axis=1).format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"
        }),
        use_container_width=True, hide_index=True
    )

    st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        p = os.path.join(RESULTS_DIR, "model_comparison.png")
        if os.path.exists(p):
            st.markdown("#### Model Comparison")
            st.image(p, use_container_width=True)
    with col2:
        p = os.path.join(RESULTS_DIR, "roc_curves.png")
        if os.path.exists(p):
            st.markdown("#### ROC Curves")
            st.image(p, use_container_width=True)

    st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)
    st.markdown("#### Radar Chart — Metrics Comparison")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors_radar = ["#00d4ff", "#ff4b6e", "#00e676", "#ffb300"]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor="#111827")
    ax.set_facecolor("#111827")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color="#c9d6e8", fontsize=9)
    ax.set_ylim(0, 1)
    ax.tick_params(colors="#5a7090")
    for spine in ax.spines.values(): spine.set_edgecolor("#1f2d45")
    ax.yaxis.set_tick_params(labelcolor="#5a7090", labelsize=7)
    ax.grid(color="#1f2d45", linewidth=0.7)

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, vals, color=colors_radar[i % len(colors_radar)], lw=2, label=row["Model"])
        ax.fill(angles, vals, color=colors_radar[i % len(colors_radar)], alpha=0.07)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8,
              labelcolor="#c9d6e8", framealpha=0, borderpad=0)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

elif page == "Confusion Matrices":
    st.markdown("# Confusion Matrices")
    st.markdown("<div class='info-box'>Each matrix shows True/False positives and negatives for Normal vs Attack classification.</div>", unsafe_allow_html=True)

    model_names = ["Decision_Tree", "Random_Forest", "Logistic_Regression", "KNN"]
    display_names = ["Decision Tree", "Random Forest", "Logistic Regression", "KNN"]

    cols = st.columns(2)
    for i, (safe, display) in enumerate(zip(model_names, display_names)):
        path = os.path.join(RESULTS_DIR, f"confusion_matrix_{safe}.png")
        with cols[i % 2]:
            st.markdown(f"#### {display}")
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info(f"Not found — run `python main.py` first.")

elif page == "Live Predictor":
    st.markdown("# Live Traffic Predictor")
    st.markdown("<div class='info-box'>Enter network connection features below. The trained model will classify the traffic as <b>Normal</b> or <b>Attack</b>.</div>", unsafe_allow_html=True)

    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH)):
        st.error("Trained model not found. Run `python main.py` to train and save the model.")
        st.stop()

    from src.predictor import DEMO_NORMAL_SAMPLE, DEMO_ATTACK_SAMPLE

    col_q1, col_q2, col_q3 = st.columns([1, 1, 4])
    with col_q1:
        load_normal = st.button("Load Normal Sample")
    with col_q2:
        load_attack = st.button("Load Attack Sample")

    if "prefill" not in st.session_state:
        st.session_state.prefill = {}
    if load_normal:
        st.session_state.prefill = DEMO_NORMAL_SAMPLE
    if load_attack:
        st.session_state.prefill = DEMO_ATTACK_SAMPLE

    pf = st.session_state.prefill

    st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)
    st.markdown("#### Connection Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        protocol_type = st.selectbox("Protocol type", ["tcp", "udp", "icmp"],
                                    index=["tcp","udp","icmp"].index(pf.get("protocol_type","tcp")))
        src_bytes = st.number_input("src_bytes", min_value=0, value=int(pf.get("src_bytes", 0)))
        dst_bytes = st.number_input("dst_bytes", min_value=0, value=int(pf.get("dst_bytes", 0)))
        duration = st.number_input("duration", min_value=0, value=int(pf.get("duration", 0)))
        count = st.number_input("count", min_value=0, value=int(pf.get("count", 1)))
        srv_count = st.number_input("srv_count", min_value=0, value=int(pf.get("srv_count", 1)))

    with c2:
        service_opts = ["http","ftp","smtp","ssh","dns","ftp_data","other",
                        "private","telnet","domain_u","ecr_i","finger","pop_3"]
        svc_default = pf.get("service", "http")
        if svc_default not in service_opts: service_opts.append(svc_default)
        service = st.selectbox("service", service_opts,
                                index=service_opts.index(svc_default))

        flag_opts = ["SF","S0","REJ","RSTO","RSTR","SH","S1","S2","S3","OTH"]
        flag_default = pf.get("flag", "SF")
        flag = st.selectbox("flag", flag_opts,
                            index=flag_opts.index(flag_default) if flag_default in flag_opts else 0)

        logged_in = st.selectbox("logged_in", [0, 1], index=int(pf.get("logged_in", 0)))
        land = st.selectbox("land", [0, 1], index=int(pf.get("land", 0)))
        hot = st.number_input("hot", min_value=0, value=int(pf.get("hot", 0)))
        num_compromised = st.number_input("num_compromised", min_value=0, value=int(pf.get("num_compromised", 0)))

    with c3:
        serror_rate = st.slider("serror_rate", 0.0, 1.0, float(pf.get("serror_rate", 0.0)), 0.01)
        rerror_rate = st.slider("rerror_rate", 0.0, 1.0, float(pf.get("rerror_rate", 0.0)), 0.01)
        same_srv_rate = st.slider("same_srv_rate", 0.0, 1.0, float(pf.get("same_srv_rate", 1.0)), 0.01)
        diff_srv_rate = st.slider("diff_srv_rate", 0.0, 1.0, float(pf.get("diff_srv_rate", 0.0)), 0.01)
        dst_host_count = st.number_input("dst_host_count", min_value=0, max_value=255, value=int(pf.get("dst_host_count", 1)))
        dst_host_srv_count = st.number_input("dst_host_srv_count", min_value=0, max_value=255, value=int(pf.get("dst_host_srv_count", 1)))

    st.markdown("<hr class='nids-divider'>", unsafe_allow_html=True)

    if st.button("🔍 Analyse Traffic"):
        sample = {
            "duration": duration, "protocol_type": protocol_type, "service": service,
            "flag": flag, "src_bytes": src_bytes, "dst_bytes": dst_bytes,
            "land": land, "wrong_fragment": 0, "urgent": 0, "hot": hot,
            "num_failed_logins": 0, "logged_in": logged_in,
            "num_compromised": num_compromised, "root_shell": 0,
            "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
            "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0,
            "is_host_login": 0, "is_guest_login": 0,
            "count": count, "srv_count": srv_count,
            "serror_rate": serror_rate, "srv_serror_rate": serror_rate,
            "rerror_rate": rerror_rate, "srv_rerror_rate": rerror_rate,
            "same_srv_rate": same_srv_rate, "diff_srv_rate": diff_srv_rate,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": dst_host_count,
            "dst_host_srv_count": dst_host_srv_count,
            "dst_host_same_srv_rate": same_srv_rate,
            "dst_host_diff_srv_rate": diff_srv_rate,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": serror_rate,
            "dst_host_srv_serror_rate": serror_rate,
            "dst_host_rerror_rate": rerror_rate,
            "dst_host_srv_rerror_rate": rerror_rate,
        }

        with st.spinner("Running inference..."):
            try:
                prep = load_preprocessor()
                model = load_model()
                from src.predictor import IntrusionPredictor
                selected_features = [
                    'flag', 'same_srv_rate', 'logged_in', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'srv_serror_rate', 'diff_srv_rate',
                    'protocol_type', 'serror_rate', 'dst_host_same_src_port_rate',
                    'dst_host_diff_srv_rate', 'service', 'dst_host_srv_diff_host_rate',
                    'count', 'dst_host_rerror_rate', 'dst_host_count', 'dst_host_srv_rerror_rate',
                    'dst_host_srv_serror_rate', 'srv_count', 'dst_host_serror_rate'
                ]
                predictor = IntrusionPredictor(MODEL_PATH, prep, selected_features)
                result = predictor.predict_one(sample)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        pred = result["prediction"]
        conf = result["confidence"]

        res_col, detail_col = st.columns([1, 1])
        with res_col:
            if pred == "Attack":
                st.markdown(f"<div class='pred-attack'>⚠ ATTACK DETECTED<br><small style='font-size:14px'>Confidence: {conf:.1%}</small></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='pred-normal'>✓ NORMAL TRAFFIC<br><small style='font-size:14px'>Confidence: {conf:.1%}</small></div>", unsafe_allow_html=True)

        with detail_col:
            if conf is not None:
                fig, ax = plt.subplots(figsize=(4, 1.2), facecolor="#111827")
                ax.set_facecolor("#111827")
                color = "#ff4b6e" if pred == "Attack" else "#00e676"
                ax.barh([""], [conf], color=color, height=0.4)
                ax.barh([""], [1 - conf], left=[conf], color="#1f2d45", height=0.4)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(["0%","25%","50%","75%","100%"], color="#5a7090", fontsize=8)
                ax.tick_params(left=False, labelleft=False)
                ax.set_title("Confidence", color="#c9d6e8", fontsize=9, pad=6)
                for spine in ax.spines.values(): spine.set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with st.expander("View submitted features"):
            st.json(sample)
