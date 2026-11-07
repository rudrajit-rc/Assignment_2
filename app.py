"""Interactive Streamlit interface for evaluating the saved classifiers."""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model_utils import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    MODEL_SPECS,
    PROJECT_DIR,
    TARGET_COLUMN,
    evaluate_predictions,
    positive_class_scores,
    validate_test_data,
)


st.set_page_config(page_title="Cell Pattern Classifier Lab", page_icon="🧪", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(35, 108, 139, 0.18), transparent 36%),
            #0b1220;
        color: #e2e8f0;
    }
    h2, h3, p, label, li, [data-testid="stCaptionContainer"] { color: #e2e8f0; }
    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    .hero {
        padding: 2rem 2.2rem;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(16, 42, 67, 0.98), rgba(15, 23, 42, 0.92));
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.2);
    }
    .hero h1 {
        margin: 0.45rem 0 0.45rem;
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 3.2rem);
        letter-spacing: -0.045em;
    }
    .hero p {
        max-width: 720px;
        margin: 0;
        color: #cbd5e1;
        font-size: 1.04rem;
        line-height: 1.65;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed rgba(103, 232, 249, 0.55);
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.52);
    }
    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.66);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
        <h1>Cell Pattern Classifier Lab</h1>
        <p>Compare five trained classifiers on your labeled data. Upload a test CSV to reveal model insights, performance metrics, and detailed predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning("Academic demonstration only. This application is not a medical diagnostic tool.")


@st.cache_resource
def load_saved_models() -> dict[str, dict]:
    artifacts = {}
    for display_name, filename in MODEL_SPECS.items():
        path = PROJECT_DIR / "model" / f"{filename}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        artifacts[display_name] = joblib.load(path)
    return artifacts


with st.container(border=True):
    upload_column, guide_column = st.columns([1.45, 1])
    with upload_column:
        uploaded_file = st.file_uploader(
            "Upload a labeled test CSV",
            type=["csv"],
            help="Include all 30 numeric features and a diagnosis column containing B or M.",
        )
    with guide_column:
        st.markdown("#### What you'll need")
        st.markdown(
            "A CSV with **30 numeric feature columns** and a **`diagnosis`** column containing `B` and `M` labels."
        )
        st.caption("Your uploaded data is used for evaluation only. The saved models are never retrained.")

if uploaded_file is None:
    st.info("Upload a valid CSV above to unlock the model comparison and generated insights.")
    with st.expander("View required feature names"):
        st.code(", ".join(FEATURE_COLUMNS + [TARGET_COLUMN]), language="text")
    st.stop()

try:
    source_frame = pd.read_csv(uploaded_file)
    test = validate_test_data(source_frame)
except (OSError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Could not use this dataset: {error}")
    st.stop()

try:
    models = load_saved_models()
except FileNotFoundError:
    st.error("Saved model files are missing. From this folder, run `python train_models.py` once, then restart the app.")
    st.stop()

st.caption(
    f"Evaluating **{len(test):,} rows** from **{uploaded_file.name}** · "
    f"{int((test[TARGET_COLUMN] == 'B').sum()):,} benign · "
    f"{int((test[TARGET_COLUMN] == 'M').sum()):,} malignant"
)

all_metrics = {}
all_predictions = {}
X_test = test[FEATURE_COLUMNS]
y_test = test[TARGET_COLUMN]
for display_name, artifact in models.items():
    model = artifact["model"]
    prediction = model.predict(X_test)
    scores = positive_class_scores(model, X_test)
    all_predictions[display_name] = prediction
    all_metrics[display_name] = evaluate_predictions(y_test, prediction, scores)

st.subheader("Insights from your upload")
recall_winner = max(all_metrics, key=lambda name: all_metrics[name]["Recall"])
auc_winner = max(all_metrics, key=lambda name: all_metrics[name]["AUC"])
accuracy_winner = max(all_metrics, key=lambda name: all_metrics[name]["Accuracy"])
insight_columns = st.columns(3)
insights = [
    ("Highest malignant recall", recall_winner, all_metrics[recall_winner]["Recall"]),
    ("Best class separation (AUC)", auc_winner, all_metrics[auc_winner]["AUC"]),
    ("Highest accuracy", accuracy_winner, all_metrics[accuracy_winner]["Accuracy"]),
]
for column, (label, model_name, score) in zip(insight_columns, insights):
    column.metric(label, f"{score:.3f}", help=f"Top model for this metric: {model_name}")
    column.caption(model_name)

st.caption(
    "Insights are calculated from the uploaded labeled rows. Malignant (M) is treated as the positive class."
)

st.subheader("Model comparison")
comparison = pd.DataFrame.from_dict(all_metrics, orient="index")
comparison.index.name = "Model"
table_column, chart_column = st.columns([1.1, 0.9])
with table_column:
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
with chart_column:
    st.bar_chart(comparison[["AUC", "Recall", "F1"]], height=300)

st.divider()
st.subheader("Explore a model")
selected_model = st.selectbox("Choose a model for detailed results", list(MODEL_SPECS))
metrics = all_metrics[selected_model]
metric_columns = st.columns(6)
for column, metric_name in zip(metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    column.metric(metric_name, f"{metrics[metric_name]:.3f}")

left, right = st.columns([1, 1])
with left:
    st.markdown("#### Confusion matrix")
    matrix = confusion_matrix(y_test, all_predictions[selected_model], labels=CLASS_LABELS)
    matrix_frame = pd.DataFrame(
        matrix,
        index=["Actual B (benign)", "Actual M (malignant)"],
        columns=["Predicted B (benign)", "Predicted M (malignant)"],
    )
    st.dataframe(matrix_frame, use_container_width=True)
with right:
    st.markdown("#### Classification report")
    report = classification_report(
        y_test,
        all_predictions[selected_model],
        labels=CLASS_LABELS,
        target_names=["Benign (B)", "Malignant (M)"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

with st.expander("CSV format and feature names"):
    st.write("The CSV must contain 30 numeric feature columns and a `diagnosis` column with B/M labels. Extra columns are ignored.")
    st.code(", ".join(FEATURE_COLUMNS + [TARGET_COLUMN]), language="text")
