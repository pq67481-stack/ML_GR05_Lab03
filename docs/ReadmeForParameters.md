# Giải thích tham số của bài Lab 03

Tài liệu này giải thích các tham số khi phân cụm khách hàng bằng K-means. Cấu hình mặc định là điểm khởi đầu; kết quả lựa chọn K trên validation được ghi ở mục 6. Thiết lập chia dữ liệu được ghi ở mục 5.

## 1. Đầu vào và đầu ra của mô hình

- **Input (đầu vào):** ba đặc trưng số đã được đưa về thang đo phù hợp: tuổi (`Age`), thu nhập hằng năm (`Annual Income`) và điểm chi tiêu (`Spending Score`).
- **Output (đầu ra):** mã cụm được gán cho mỗi khách hàng và các tâm cụm được học từ dữ liệu. Mã cụm chỉ là tên nhóm, không thể hiện thứ hạng.
- **Mục đích:** tìm những khách hàng có đặc điểm gần nhau. Mô hình không có nhãn cụm đúng sẵn để học.

## 2. Siêu tham số của K-means

**Hyperparameter (siêu tham số)** là giá trị được chọn trước khi huấn luyện, ảnh hưởng tới cách thuật toán hoạt động. Các tên trong bảng đã được cài đặt trong `src/model.py`.

| Tên dự kiến | Ý nghĩa và tác dụng | Cách lựa chọn hoặc kiểm tra |
|---|---|---|
| `n_clusters` | Số cụm K cần tìm. Ví dụ K = 3 yêu cầu mô hình tạo ba nhóm. | Thử nhiều K hợp lệ; tham khảo Elbow, Silhouette và khả năng giải thích cụm. Không mặc định ví dụ K = 3 là kết quả của bài. |
| `init` | Cách chọn các tâm cụm ban đầu. | Hiện chỉ hỗ trợ `random`: lấy ngẫu nhiên K điểm có giá trị khác nhau. Chưa triển khai K-means++. |
| `n_init` | Số lần chạy với các tâm ban đầu khác nhau; chọn nghiệm tốt nhất theo inertia. | Dùng số nguyên dương và ghi rõ giá trị thực tế; nhiều lần khởi tạo thường tốn thêm thời gian. |
| `max_iter` | Số vòng lặp tối đa cho mỗi lần khởi tạo. | Dùng số nguyên dương; kiểm tra mô hình dừng do hội tụ hay do chạm giới hạn. |
| `tol` | Ngưỡng dung sai dùng trong điều kiện dừng khi tâm cụm thay đổi rất ít. | Dùng giá trị không âm; định nghĩa chính xác độ thay đổi cần được ghi theo cách triển khai. |
| `random_state` | Hạt giống sinh số ngẫu nhiên, giúp lặp lại cùng quá trình khởi tạo trong cùng điều kiện. | Ghi rõ giá trị sử dụng; cùng seed không đảm bảo kết quả giống nhau nếu dữ liệu hoặc cách triển khai thay đổi. |

## 3. Tham số học được và kết quả huấn luyện

**Parameter (tham số mô hình)** là giá trị mô hình học từ dữ liệu. Với K-means, đó là tọa độ các **centroid (tâm cụm)** trong không gian đặc trưng đã xử lý. Ví dụ, mỗi tâm của bài này có ba tọa độ tương ứng với ba đặc trưng.

Ngoài ra có thể lưu:

- `inertia`: tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm được gán. Giá trị nhỏ hơn không tự chứng minh phân cụm tốt hơn vì inertia thường giảm khi tăng K.
- Số vòng lặp đã thực hiện và điều kiện dừng: giúp kiểm tra quá trình huấn luyện.
- Tham số tiền xử lý học từ dữ liệu huấn luyện, ví dụ trung bình và độ lệch chuẩn của từng cột nếu dùng standardization (chuẩn hóa theo trung bình và độ lệch chuẩn). Các thông số này cần được lưu cùng mô hình để xử lý dữ liệu mới nhất quán.

## 4. Tham số liên quan đến dữ liệu và đánh giá

| Nội dung | Ý nghĩa | Lưu ý |
|---|---|---|
| Tỷ lệ chia dữ liệu | Tỷ lệ mẫu dành cho train (huấn luyện), validation (chọn cấu hình) và test (đánh giá cuối), nếu chọn chia tập. | Chia trước khi học thông số chuẩn hóa; chỉ dùng train để tính các thông số đó. Cần giải thích lý do chia hay không chia cho bài toán phân cụm. |
| Khoảng K thử nghiệm | Tập các giá trị `n_clusters` đem ra so sánh. | Chỉ xét K hợp lệ với số mẫu; chọn trên dữ liệu dành cho lựa chọn cấu hình, không dựa vào tập test cuối. |
| Tham số lấy mẫu Silhouette | Số điểm được dùng nếu tính chỉ số trên một mẫu con. | Ghi rõ số điểm, cách lấy mẫu và seed; kiểm tra có ít nhất hai cụm và số cụm nhỏ hơn số điểm được đánh giá. |

**Elbow (phương pháp khuỷu tay)** quan sát đường inertia theo K để tìm vùng mà mức giảm bắt đầu chậm lại. **Silhouette (hệ số bóng)** đánh giá tương đối mức gần các điểm trong cùng cụm so với cụm khác. Cả hai chỉ hỗ trợ lựa chọn; cần xem thêm đặc điểm thực tế của các cụm.

## 5. Thiết lập dữ liệu đã thực hiện

- Tỷ lệ train / validation / test: 70% / 15% / phần còn lại, tương ứng 10.555 / 2.261 / 2.263 khách trong dữ liệu hiện tại.
- `random_state = 42` điều khiển việc xáo trộn chia dữ liệu. Cấu hình K-means cũng có seed riêng, hiện đặt 42; hai seed có vai trò khác nhau.
- Chuẩn hóa Z-score dùng độ lệch chuẩn tổng thể `ddof=0`; trung bình và độ lệch chuẩn được tính từ train và lưu trong `data/processed/train_scaler.json`.
- Các thông số mô hình và cách chọn K của lần thực nghiệm được trình bày ở mục 6.

## 6. Trạng thái triển khai

`config.yaml` đặt cấu hình khởi đầu: `n_clusters=3`, `init=random`, `n_init=5`, `max_iter=300`, `tol=0.0001`, `random_state=42`. `tol` đo chuẩn Euclid của toàn bộ thay đổi tọa độ các tâm giữa hai vòng. Thực nghiệm đã thử K=2–8; lựa chọn K=6 theo Silhouette validation như báo cáo. Giá trị `n_clusters=3` trong mục model là giá trị ban đầu và được thay bởi từng K khi tìm kiếm, không phải K được chọn. Mô hình lưu tại `models/kmeans.json` chứa K=6 và thông số chuẩn hóa train. Silhouette được tính chính xác trên toàn bộ tập đánh giá bằng các lô 128 điểm, không lấy mẫu.
