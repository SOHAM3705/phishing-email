"""Streamlit interface for the offline phishing URL detector."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from phishing_url_detector import analyse_url, train_and_evaluate


DEFAULT_DATASET = Path(__file__).with_name("phish.csv")

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(show_spinner="Training the Random Forest model...")
def load_model(dataset_path, modified_time, file_size):
    return train_and_evaluate(dataset_path)


st.title("Phishing URL Detector")
st.caption(
    "Assess URL risk from its text alone. The scanner never opens URLs "
    "or makes network requests."
)

with st.sidebar:
    st.header("Model dataset")
    dataset_input = st.text_input("CSV path", value=str(DEFAULT_DATASET))
    st.caption("The CSV must contain `url` and `status` columns.")
    page = st.radio(
        "Navigation",
        ["Model overview", "URL scanner", "Project details"],
    )

dataset_path = Path(dataset_input).expanduser().resolve()
if not dataset_path.is_file():
    st.error(f"Dataset file not found: {dataset_path}")
    st.stop()

dataset_stat = dataset_path.stat()
model, columns, metrics, (n_train, n_test) = load_model(
    str(dataset_path), dataset_stat.st_mtime_ns, dataset_stat.st_size
)

if page == "Model overview":
    st.subheader("Test-set performance")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
    metric_columns[1].metric("Precision", f"{metrics['precision']:.2%}")
    metric_columns[2].metric("Recall", f"{metrics['recall']:.2%}")
    metric_columns[3].metric("F1 score", f"{metrics['f1_score']:.2%}")

    st.caption(
        f"Random Forest · 200 trees · {n_train:,} training rows · "
        f"{n_test:,} test rows · {len(columns)} lexical features"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        confusion = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual legitimate", "Actual phishing"],
            columns=["Predicted legitimate", "Predicted phishing"],
        )
        st.dataframe(confusion, use_container_width=True)
    with right:
        st.markdown("#### Most influential URL features")
        importance = pd.Series(
            model.feature_importances_, index=columns, name="Importance"
        ).nlargest(10).sort_values()
        importance_chart = importance.rename_axis("Feature").reset_index()
        st.bar_chart(
            importance_chart,
            x="Importance",
            y="Feature",
            horizontal=True,
        )

elif page == "URL scanner":
    st.subheader("Analyze one or more URLs")
    st.info(
        "Paste one URL per line. Analysis uses URL text only; links are not "
        "visited, resolved, or sent to a third party."
    )
    with st.form("url_scan"):
        urls_text = st.text_area(
            "URLs",
            placeholder=(
                "https://www.example.com/account\n"
                "http://192.0.2.1/login/verify"
            ),
            height=150,
        )
        submitted = st.form_submit_button("Analyze URLs", type="primary")

    if submitted:
        urls = [line.strip() for line in urls_text.splitlines() if line.strip()]
        if not urls:
            st.warning("Enter at least one URL to analyze.")
        else:
            results = [analyse_url(model, columns, url) for url in urls]
            summary = pd.DataFrame(
                [
                    {
                        "URL": result["url"],
                        "Prediction": result["prediction"].title(),
                        "Phishing probability": (
                            f"{result['phishing_probability']:.1%}"
                        ),
                        "Risk": result["risk_level"],
                    }
                    for result in results
                ]
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)

            for index, result in enumerate(results, start=1):
                with st.expander(f"{index}. {result['url']}"):
                    st.write(
                        f"**Prediction:** {result['prediction'].title()}  \n"
                        f"**Phishing probability:** "
                        f"{result['phishing_probability']:.1%}  \n"
                        f"**Risk level:** {result['risk_level']}"
                    )
                    if result["indicators"]:
                        st.markdown("**Suspicious indicators**")
                        for indicator in result["indicators"]:
                            st.markdown(f"- {indicator}")
                    else:
                        st.write("No rule-based indicators were detected.")

            report = {
                "model": "RandomForest (200 trees)",
                "dataset_size": n_train + n_test,
                "metrics": metrics,
                "url_reports": results,
            }
            st.download_button(
                "Download JSON report",
                data=json.dumps(report, indent=4),
                file_name="phishing_url_report.json",
                mime="application/json",
            )

else:
    st.subheader("How the detector works")
    st.markdown(
        """
        1. Load labeled URLs from the selected CSV (`url` and `status`).
        2. Extract 29 lexical features from each URL string.
        3. Train and evaluate a 200-tree Random Forest with a stratified 70/30 split.
        4. Report phishing probability, classification, risk level, and readable indicators.
        5. Export model metrics and URL findings as JSON.

        **Risk thresholds:** LOW below 50%, MEDIUM from 50% to below 80%, and
        HIGH at 80% or above. A URL is classified as phishing at 50% or above.

        **Limitations:** The detector only examines URL text. It cannot inspect
        page contents or confirm whether a live site is malicious, and test-set
        performance is not a guarantee of real-world accuracy.
        """
    )
