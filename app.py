"""Giao diện Streamlit gán cụm cho một khách hàng mới."""

from pathlib import Path

import streamlit as st

from src.predict import predict_customer


def main():
    """Nhận đặc trưng, nạp mô hình JSON và trình bày nhóm khách gần nhất."""
    st.set_page_config(page_title="Phân cụm khách hàng", page_icon="🛍️")
    st.title("Phân cụm khách hàng")
    st.caption("Lab 03 — K-means tự triển khai")
    st.write("Nhập ba đặc trưng để tìm nhóm khách hàng có tâm cụm gần nhất.")
    st.caption("Thu nhập phải cùng đơn vị với dữ liệu gốc; đơn vị tiền tệ chưa được xác minh.")
    with st.form("customer_form"):
        age = st.number_input("Tuổi", min_value=1, max_value=150, value=35, step=1)
        income = st.number_input("Thu nhập hằng năm (đơn vị dữ liệu gốc)", min_value=0.0, value=80000.0, step=1000.0)
        spending = st.number_input("Điểm chi tiêu (1–100)", min_value=1.0, max_value=100.0, value=60.0, step=1.0)
        submitted = st.form_submit_button("Dự đoán cụm", type="primary")
    if submitted:
        try:
            model_path = Path(__file__).resolve().parent / "models/kmeans.json"
            result = predict_customer(age, income, spending, model_path)
            st.success(f"Khách hàng được gán vào cụm {result['cluster']}")
            st.write("**Tâm cụm trong đơn vị gốc:**")
            center = result["center"]
            st.write(f"Tuổi: {center['Age']:.1f} · Thu nhập năm: {center['Annual Income']:,.0f} · Điểm chi tiêu: {center['Spending Score']:.1f}")
            st.caption("Mã cụm là tên nhóm, không phải thứ hạng hay nhãn đúng/sai.")
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Không thể dự đoán: {error}")


if __name__ == "__main__":
    main()
