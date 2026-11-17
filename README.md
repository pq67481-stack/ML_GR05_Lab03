# Lab 03: Phân cụm khách hàng

Dự án thực hành học không giám sát với dữ liệu khách hàng. Mục tiêu là tìm các nhóm khách hàng có đặc điểm tương đồng bằng K-means tự triển khai, sử dụng tuổi, thu nhập hằng năm và điểm chi tiêu.

## Cấu trúc hiện tại

```text
LA03/
├── AGENTS.md                 Quy tắc thực hiện dự án
├── README.md                 Giới thiệu và cấu trúc dự án
├── app.py                    Giao diện Streamlit dự đoán cụm
├── config.yaml               Cấu hình dữ liệu, mô hình và thực nghiệm
├── requirements.txt          Thư viện cần thiết
├── data/
│   ├── raw/
│   │   └── Shopping_Mall_15079.csv  Dữ liệu gốc đang sử dụng
│   └── processed/            Dữ liệu sau xử lý khi được tạo
├── src/                      Xử lý dữ liệu, mô hình, đánh giá, tracking và dự đoán
├── notebooks/                Notebook giải thích và thực nghiệm
├── models/                   Mô hình đã huấn luyện khi được lưu
├── reports/                  Bảng, kết quả và lịch sử thực nghiệm
├── logs/                     Nhật ký từng lần huấn luyện
├── progress_results/         Trạng thái gần nhất của mỗi lần huấn luyện
└── docs/
    ├── Lab03_descript.md     Báo cáo quá trình và kết quả
    ├── ReadmeForParameters.md Giải thích tham số mô hình
    └── images/               Biểu đồ phục vụ báo cáo
```

Thư mục `.venv/` có thể còn xuất hiện do tệp Python của môi trường cũ chưa xóa được; đây không phải dữ liệu hay mã nguồn bài lab.

## Trạng thái

Toàn bộ EDA, chia tập, chuẩn hóa, huấn luyện và báo cáo chính hiện dùng **15.079 dòng** trong `Shopping_Mall_15079.csv`. Tệp được tải từ [bản sao GitHub](https://github.com/fiqritiras/mall-customer-segmentation), dự án đó ghi nguồn [Kaggle](https://www.kaggle.com/datasets/zubairmustafa/shopping-mall-customer-segmentation-data). Năm dòng đầu và cuối khớp ảnh video; nguồn thu thập thực tế và đơn vị tiền tệ chưa được xác minh.

Theo lần chạy hiện tại, K=6 được chọn từ validation; Silhouette validation ≈ 0,282, test ≈ 0,292. Test được đánh giá sau khi chọn K. Chi tiết và hạn chế trong `docs/Lab03_descript.md`.

## Cách xem và chạy

1. Cài Python và thư viện: `python -m pip install -r requirements.txt`.
2. Mở `notebooks/Lab03.ipynb` trong VS Code hoặc Jupyter và chạy các ô theo thứ tự từ đầu. Lệnh cài và giao diện Jupyter chưa được kiểm tra trong phiên hiện tại; nội dung các ô đã được chạy tuần tự bằng IPython của Python 3.12.
3. Xem giải thích và biểu đồ trong `docs/Lab03_descript.md`; bảng EDA trong `reports/`; đặc trưng và thông số trong `data/processed/`.

Notebook thực hiện bước 4 chia tập, bước 5 làm sạch từng tập, bước 6 chuyển overlap về train và bước 7 chuẩn hóa theo train. Nếu có cùng mã khách nhưng nội dung mâu thuẫn, chương trình dừng để xem xét; không tự xóa. Danh sách chuyển và kết quả overlap nằm trong `reports/`.

Để đánh giá trên tập giữ riêng, dùng `train_scaled.csv`, `validation_scaled.csv`, `test_scaled.csv` và `train_scaler.json`, có thông số chỉ học từ train. `features_scaled.csv` và `scaler.json` là tệp minh họa toàn mẫu từ lần chạy cũ, không được notebook hiện tại tạo lại và không dùng để đánh giá ngoài mẫu. Các tệp đầu ra có thể bị ghi đè khi chạy lại.

Kiểm tra mô hình từ thư mục gốc: `python -m unittest discover -s tests -v`. File `config.yaml` là JSON hợp lệ trong YAML, được `src.model.load_config` đọc mà không cần thư viện YAML.

Huấn luyện và đánh giá lại từ thư mục gốc: `python -m src.train`. Lệnh này dùng các tập đã tạo bởi phần chuẩn bị dữ liệu trong notebook, ghi đè kết quả trong `reports/`, mô hình trong `models/` và hai biểu đồ trong `docs/images/`.

## Bài bổ sung phân vùng ảnh

Bài bổ sung nằm riêng tại `additional_image/`. Xem `additional_image/README.md` và `additional_image/notebooks/image_segmentation.ipynb` để chạy.

## Demo với khách hàng mới

Từ thư mục gốc, chạy giao diện bằng môi trường đã chuẩn bị trên máy:

```powershell
.\.venv_demo\Scripts\python.exe -m streamlit run app.py
```

Nhập tuổi, thu nhập hằng năm theo cùng đơn vị của CSV gốc và điểm chi tiêu 1–100. Chương trình nạp `models/kmeans.json`, áp dụng thông số chuẩn hóa của train và trả về mã cụm cùng tâm cụm. Mã cụm không phải thứ hạng; nguồn dữ liệu chưa xác minh được đơn vị tiền tệ.

Có thể dự đoán qua terminal, ví dụ dữ liệu minh họa:

```powershell
.\.venv_demo\Scripts\python.exe -X utf8 -m src.predict --age 35 --income 80000 --spending 60
```

Nếu dùng môi trường Python khác, cài `requirements.txt` rồi thay đường dẫn Python trong lệnh cho phù hợp.

## Theo dõi thực nghiệm

Mỗi lần `run_experiment` chạy qua notebook hoặc `python -m src.train`, chương trình tạo một `run_id` và ghi thời gian theo UTC:

- `logs/<run_id>.log`: các sự kiện bắt đầu, kết thúc hoặc lỗi theo từng bước.
- `progress_results/<run_id>.json`: trạng thái gần nhất của lần chạy.
- `reports/experiment_logs.csv`: một dòng tổng hợp cho mỗi lần chạy thành công.
- `reports/experiments/<run_id>/`: cấu hình, chỉ số, hash các tệp đầu vào, phiên bản thư viện, bảng thử K và bản sao mô hình. Những bản này không bị ghi đè khi chạy lại.

Các tệp kết quả dùng chung ở `reports/` và `models/kmeans.json` vẫn được cập nhật theo lần chạy mới; lịch sử chỉ được ghi từ khi tính năng tracking được thêm.
