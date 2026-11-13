# Lab 03 · Phân cụm khách hàng

Bài thực hành Chapter 3 của môn Machine Learning dùng 200 quan sát trong `Mall_Customers.csv`. Pipeline so sánh K-means với Gaussian Mixture, chọn cấu hình bằng kiểm định chéo trên train, đánh giá trên test và lưu mô hình để gán cụm cho khách mới. Bài bổ sung dùng K-means để phân đoạn ảnh theo màu. [Nguồn dữ liệu Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Chạy trên Windows

Mở PowerShell tại thư mục `LA03`. Môi trường đã dùng để kiểm chứng là Python 3.12. Nếu thư mục đã có `.venv` hoạt động, có thể chạy trực tiếp từ lệnh thứ ba.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.pipeline
.\.venv\Scripts\python.exe -m src.predict --age 35 --income 70 --spending 50
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit thường mở ở `http://localhost:8501`. Tab **Gán cụm** nhận tuổi, thu nhập theo nghìn USD và điểm chi tiêu. Tab **Kết quả thực nghiệm** hiển thị các chỉ số, biểu đồ và bảng hồ sơ cụm. Có thể dùng `--model` với CLI để chọn mô hình từ một lần chạy cụ thể. Chỉ nạp tệp Joblib từ nguồn bạn tin cậy.

### Notebook và bài ảnh

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name la03 --display-name "Python (Lab 03)"
.\.venv\Scripts\python.exe -m notebook notebooks/lab03.ipynb
.\.venv\Scripts\python.exe -m src.image_segmentation
.\.venv\Scripts\python.exe -m src.image_segmentation --image "C:\duong-dan\anh.jpg" --k 3 6 9 --max-side 320
```

Chọn kernel **Python (Lab 03)** và chạy **Run All**. Notebook tạo một lần chạy phân cụm mới và thực hiện bài ảnh. Lệnh ảnh mặc định dùng ảnh mẫu Grace Hopper đi kèm Matplotlib. Lần chạy ảnh mới cập nhật kết quả ở `reports/image_segmentation/` và `docs/images/image_segmentation/`.

## Cấu trúc dự án

```text
LA03/
├── README.md, requirements.txt, config.yaml, app.py
├── data/
│   ├── raw/Mall_Customers.csv
│   └── processed/<run_id>/{train,test}.csv
├── src/
│   ├── pipeline.py, eda.py, preprocess.py
│   ├── model.py, train.py, evaluate.py, predict.py
│   └── visualize.py, report.py, tracking.py, image_segmentation.py
├── models/<run_id>/customer_clusters.joblib
├── notebooks/lab03.ipynb
├── reports/<run_id>/ và reports/image_segmentation/
├── docs/
│   ├── Lab_03_descript.md
│   ├── ReadmeForParameters.md
│   └── images/<run_id>/ và images/image_segmentation/
├── logs/<run_id>.log
└── progress_results/<run_id>.json và latest.json
```

Mỗi lần chạy khách hàng có một ID riêng. `latest.json` trỏ tới lần chạy hoàn thành gần nhất. Mô hình và báo cáo luôn dùng cùng một ID. CSV train và test lưu giá trị gốc, đặc trưng chuẩn hóa, cụm được gán và Silhouette từng khách. `experiment.json` lưu bản chụp cấu hình, phiên bản chính và SHA-256 của tệp dữ liệu gốc.

## Đọc kết quả

- [Quy trình, công thức và diễn giải chi tiết](docs/Lab_03_descript.md).
- [Giải thích tham số](docs/ReadmeForParameters.md).
- [Báo cáo số liệu lần chạy tham chiếu](reports/20261112T160608Z-052b5864/results.md).
- [Notebook Lab 03](notebooks/lab03.ipynb).

Ở cấu hình mặc định, **K-means với 4 cụm** đạt Silhouette trung bình **0,378** trên kiểm định chéo và **0,383** trên 40 khách test. Đây là chỉ số hình học của phân cụm. Tài liệu phân tích giải thích vì sao hai chỉ số phụ ưu tiên số cụm khác và kết quả này phù hợp đến đâu với việc mô tả khách hàng.
