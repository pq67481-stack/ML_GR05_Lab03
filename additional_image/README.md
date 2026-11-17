# Lab 03 bổ sung — Phân vùng ảnh bằng K-means

Bài thực hành nhóm các pixel theo ba đặc trưng màu RGB. Ảnh minh họa `data/raw/grace_hopper.jpg` được sao chép từ `matplotlib/mpl-data/sample_data/grace_hopper.jpg` của bản Matplotlib trên máy. Xem nguồn gốc ảnh và điều khoản sử dụng trước khi tái phân phối ngoài mục đích bài tập.

```text
additional_image/
├── config.yaml                Tham số ảnh và mô hình
├── requirements.txt           Thư viện cho bài bổ sung
├── data/raw/                  Ảnh gốc
├── src/                       Chuẩn bị ảnh và vẽ kết quả
├── notebooks/                 Notebook thực hành
├── results/                   Ảnh PNG và metrics.json
└── docs/                      Báo cáo
```

## Chạy notebook

Từ thư mục gốc LA03, cài thư viện trong môi trường Python của bạn bằng `python -m pip install -r additional_image/requirements.txt`. Mở `additional_image/notebooks/image_segmentation.ipynb` bằng VS Code hoặc Jupyter rồi chạy từ trên xuống. Notebook cần thư mục `src/model.py` của dự án chính vì thuật toán K-means được dùng chung.

Trên máy hiện tại có thể chạy Jupyter bằng `.\.venv_demo\Scripts\python.exe -m notebook`. Lệnh mở giao diện Jupyter chưa được kiểm tra trong phiên này. Các ô notebook đã được chạy tuần tự bằng IPython.

Thay `image_path` trong `config.yaml` để dùng ảnh khác nằm trong thư mục `additional_image/`. Cấu hình dùng cú pháp JSON hợp lệ trong YAML, đọc bằng thư viện chuẩn. `train_max_side` là cạnh tối đa của ảnh nhỏ dùng để fit; `prediction_batch_size` giới hạn số pixel gốc được predict mỗi lần. Kết quả cùng tên sẽ bị ghi đè khi chạy lại.

Xem `results/comparison.png`, ảnh riêng `segmented_k*.png`, tâm màu và số vòng tại `results/metrics.json`. Báo cáo nằm trong `docs/Lab03_additional.md`.
