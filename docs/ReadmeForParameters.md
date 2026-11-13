# Tham số thực nghiệm Lab 03

## Dữ liệu và thiết kế thí nghiệm

| Khóa trong `config.yaml` | Mặc định | Vai trò |
|---|---|---|
| `data.path` | `data/raw/Mall_Customers.csv` | Đường dẫn tương đối so với thư mục dự án hoặc đường dẫn tuyệt đối. |
| `data.test_size` | `0.20` | Giữ riêng 20% khách để đánh giá mô hình đã chọn. Khi đổi tỷ lệ, giữ đủ điểm trong từng fold để tính Silhouette. |
| `data.seed` | `42` | Cố định phép chia dữ liệu, xáo trộn fold, khởi tạo và dãy seed dùng khi kiểm tra độ ổn định. |
| `features` | Tuổi, thu nhập hằng năm, điểm chi tiêu | Ba biến số của bài. Thứ tự cột được lưu cùng mô hình. |
| `training.folds` | `5` | Số fold của KFold có xáo trộn, chỉ chạy trên train. Với 160 khách, mỗi lượt có 128 khách fit và 32 khách validation. |
| `training.cluster_counts` | `[2, 3, 4, 5, 6, 7, 8, 9, 10]` | Lưới số cụm/thành phần. `k=1` nằm ngoài miền xác định của Silhouette. |
| `stability.repeats` | `30` | Số lần lấy mẫu lại để đo ARI trên cùng 40 khách test. |
| `stability.fraction` | `0.80` | Lấy 80% train không hoàn lại trong mỗi lượt. Tăng tỷ lệ thường làm các mẫu fit giống train gốc hơn. |

## K-means

| Khóa | Mặc định | Tác động |
|---|---|---|
| `init` | `k-means++` | Phân tán các tâm khởi tạo để giảm khả năng đặt nhiều tâm ban đầu trong cùng một vùng. |
| `n_init` | `30` | Thử 30 lần khởi tạo cho mỗi lần fit và giữ nghiệm có inertia thấp nhất. Tăng số lần có thể giảm ảnh hưởng của khởi tạo với chi phí tính toán cao hơn. |
| `max_iter` | `300` | Giới hạn vòng cập nhật trong mỗi lần khởi tạo. |

K-means chọn tâm gần nhất trong không gian chuẩn hóa và tối thiểu hóa tổng bình phương khoảng cách tới tâm. Inertia dùng nội bộ để chọn nghiệm khởi tạo cho cùng `k`. Khi tăng `k`, nghiệm tối ưu có thể giảm inertia chỉ nhờ có thêm tâm, nên inertia thô không được dùng để xếp hạng các giá trị `k`. [Tài liệu KMeans](https://scikit-learn.org/1.7/modules/generated/sklearn.cluster.KMeans.html)

## Gaussian Mixture

| Khóa | Mặc định | Tác động |
|---|---|---|
| `covariance_type` | `full` | Mỗi thành phần có ma trận hiệp phương sai riêng, cho phép cụm có độ dài và hướng khác nhau. Cấu hình này cần ước lượng nhiều tham số hơn `diag`. |
| `n_init` | `15` | Thử nhiều khởi tạo EM và giữ nghiệm có cận dưới log-likelihood cao nhất. |
| `max_iter` | `300` | Giới hạn vòng lặp EM cho mỗi khởi tạo. |
| `reg_covar` | `0.000001` | Cộng một giá trị dương vào đường chéo hiệp phương sai để giảm nguy cơ ma trận suy biến. |

Để tính các chỉ số chung, GMM gán nhãn cứng theo thành phần có xác suất hậu nghiệm lớn nhất. Pipeline yêu cầu Silhouette xác định ở cả 5 fold và GMM hội tụ ở cả 5 fold. Cột `converged_folds` của K-means ghi số lượt fit hoàn tất, do API K-means không công bố cờ `converged_` như GMM. [Tài liệu GaussianMixture](https://scikit-learn.org/1.7/modules/generated/sklearn.mixture.GaussianMixture.html)

## Bài phân đoạn ảnh

`--k` nhận một hoặc nhiều số cụm màu. `--max-side` là giới hạn cạnh dài nhất của ảnh dùng để fit. Cạnh lớn hơn giữ nhiều chi tiết hơn và tăng số pixel cần xử lý. Phép tái tạo dùng tâm cụm đã làm tròn về RGB 8-bit và nội suy láng giềng gần nhất để duy trì bảng màu ở kích thước xuất.
