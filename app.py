import json

import pandas as pd
import streamlit as st

from src.model import load_model
from src.predict import assign_customer
from src.preprocess import AGE, INCOME, ROOT, SPEND
from src.tracking import latest_run
from src.visualize import FAMILY_NAMES

st.set_page_config(page_title="Lab 03 · Phân cụm khách hàng", layout="wide")
st.title("Phân cụm khách hàng")

try:
    run = latest_run()
    reports = ROOT / run["paths"]["reports"]
    images = ROOT / run["paths"]["images"]
    bundle = load_model(ROOT / run["paths"]["model"] / "customer_clusters.joblib")
    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    profiles = pd.read_csv(reports / "train_profile.csv")
except FileNotFoundError:
    st.info("Chạy `python -m src.pipeline` để tạo mô hình và báo cáo.")
    st.stop()

st.caption(f"{FAMILY_NAMES[bundle['family']]} · {bundle['k']} cụm · "
           f"{metrics['train_rows']} khách train / {metrics['test_rows']} khách test")
predict_tab, results_tab = st.tabs(["Gán cụm", "Kết quả thực nghiệm"])

with predict_tab:
    form_col, result_col = st.columns([3, 2], gap="large")
    with form_col:
        with st.form("customer"):
            age = st.number_input("Tuổi", min_value=1.0, value=35.0, step=1.0)
            income = st.number_input("Thu nhập hằng năm (nghìn USD)", min_value=0.0, value=70.0)
            spending = st.slider("Điểm chi tiêu", 1, 100, 50)
            submitted = st.form_submit_button("Gán cụm", type="primary")
    with result_col:
        st.subheader("Cụm của khách hàng")
        if submitted:
            prediction = assign_customer(bundle, age, income, spending)
            cluster = prediction["cluster"]
            st.metric("Cụm được gán", cluster)
            matches = profiles.loc[profiles["cluster"].eq(cluster)]
            if matches.empty:
                st.info("Cụm này chưa có khách được gán nhãn cứng trong tập train.")
            else:
                selected = matches.iloc[0]
                st.write(f"{int(selected['count'])} khách thuộc cụm này trong tập train.")
                st.dataframe(pd.DataFrame({
                    "Chỉ số": ["Tuổi", "Thu nhập (nghìn USD)", "Điểm chi tiêu"],
                    "Trung bình cụm": [selected[f"{name}_mean"] for name in (AGE, INCOME, SPEND)],
                }).round(1), hide_index=True)
            if prediction["outside_train_range"]:
                st.warning("Ngoài khoảng quan sát trên train: " + ", ".join(prediction["outside_train_range"]))
        else:
            st.caption("Nhập thông tin khách hàng để xem cụm và hồ sơ trung bình của cụm đó.")

with results_tab:
    metrics_columns = st.columns(4)
    for column, label, key in zip(
        metrics_columns[:3], ["Silhouette test ↑", "Davies–Bouldin test ↓", "Calinski–Harabasz test ↑"],
        ["silhouette", "davies_bouldin", "calinski_harabasz"], strict=True,
    ):
        score = metrics["test"][key]
        column.metric(label, f"{score:.3f}" if score is not None else "—")
    metrics_columns[3].metric("ARI trung bình ↑", f"{metrics['stability']['mean_ari']:.3f}")
    st.image(str(images / "segments.png"), width="stretch")
    st.image(str(images / "cv_metrics.png"), width="stretch")
    st.image(str(images / "silhouette.png"), width="stretch")
    st.subheader("Hồ sơ cụm trên tập train")
    st.dataframe(profiles.round(2), hide_index=True)
    st.download_button("Tải bảng so sánh cấu hình", (reports / "cv_summary.csv").read_bytes(),
                       file_name="cv_summary.csv", mime="text/csv")
