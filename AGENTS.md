# Quy tắc làm việc cho Lab 03: Phân cụm khách hàng

Áp dụng cho toàn bộ dự án LA03. Đọc yêu cầu mới nhất của người dùng và nội dung đề trước khi thực hiện; nếu có khác biệt, ưu tiên yêu cầu cụ thể mới nhất của người dùng. Quy tắc này dựa trên dự án mẫu Lab 01 và được điều chỉnh cho bài học không giám sát.

## 1. Phạm vi và mục tiêu

- Thực hiện bài phân nhóm khách hàng cửa hàng bán lẻ theo đặc điểm tương đồng, nhằm tìm và mô tả các nhóm khách hàng khác biệt.
- Dữ liệu gốc hiện dùng: `data/raw/Shopping_Mall_15079.csv`; giữ nguyên dữ liệu gốc, ghi rõ nguồn tải và các giới hạn về nguồn thu thập.
- Đặc trưng chính theo đề: `Age` (tuổi), `Annual Income` (thu nhập hằng năm, đơn vị gốc chưa được xác minh) và `Spending Score` (điểm chi tiêu). `Customer ID` được đổi tên nội bộ thành `CustomerID` để truy vết, không đưa vào khoảng cách; `Gender` có thể dùng để mô tả cụm.
- Bài này là **unsupervised learning** (học không giám sát): dữ liệu không có nhãn cụm đúng để huấn luyện hoặc xác nhận dự đoán. Số hiệu cụm chỉ là mã nhóm, không phải thứ hạng.
- Thuật toán chính: K-means, tự triển khai các bước cốt lõi bằng Python và NumPy; không dùng triển khai K-means hoặc các chỉ số phân cụm có sẵn của scikit-learn thay cho phần cần tự làm. Các thư viện xử lý bảng và vẽ biểu đồ được dùng khi cần.
- Chỉ làm bài dữ liệu khách hàng. Chưa triển khai bài ảnh cho đến khi người dùng yêu cầu.

## 2. Cách làm việc

Với mỗi yêu cầu: đọc hiện trạng và xác định phạm vi; lập kế hoạch phù hợp với độ lớn công việc; triển khai phần được giao; chạy kiểm tra phù hợp; cập nhật giải thích, báo cáo và biểu đồ liên quan; cuối cùng thông báo kết quả thực tế và giới hạn. Có thể thông báo kế hoạch ngắn trước khi thực hiện; không dừng xin xác nhận cho lựa chọn thông thường đã được người dùng cho phép.

Ưu tiên dự án phục vụ học tập: dễ đọc, dễ kiểm chứng, không thiết kế phức tạp, không tạo hệ thống MLOps hoặc tự động hóa không cần thiết. Không tự triển khai các phần ngoài phạm vi yêu cầu hiện tại. Khi được yêu cầu hoàn thành toàn bộ một nhiệm vụ, tiếp tục đến khi hoàn tất các bước cần thiết.

## 3. Quy trình bài lab

Thực hiện theo thứ tự, giải thích mục đích, đầu vào, xử lý, đầu ra và cách kiểm tra của mỗi bước quan trọng:

1. **Problem Definition:** xác định mục tiêu, dữ liệu, đầu ra, giả định và giới hạn của bài toán.
2. **Data Understanding / EDA:** kiểm tra kích thước, kiểu, giá trị thiếu, trùng lặp, phân phối, ngoại lệ và quan hệ giữa đặc trưng. Vẽ histogram hoặc scatter plot phù hợp. Giá trị ngoại lệ được gắn cờ để xem xét, không mặc định xóa.
3. **Data Preparation:** giải thích lựa chọn đặc trưng, xử lý dữ liệu thiếu hoặc trùng nếu có, và chuẩn hóa hoặc đưa về cùng thang đo. Nếu chia train, validation và test thì chia trước; chỉ học tham số chuẩn hóa trên train, sau đó áp dụng cho các tập còn lại để tránh rò rỉ dữ liệu. Nêu lý do chọn hoặc không chọn cách chia khi thực nghiệm không giám sát.
4. **Model Understanding:** trình bày đầu vào, đầu ra, tâm cụm, khoảng cách, hàm mục tiêu inertia, tham số học được và siêu tham số; giải thích công thức và giới hạn.
5. **Model Implementation:** tự cài đặt khởi tạo tâm, tính khoảng cách, gán cụm, cập nhật tâm, xử lý cụm rỗng và điều kiện dừng; liên hệ từng phép tính với code.
6. **Training & Inference:** cấu hình, huấn luyện, thử các giá trị K hợp lý và gán cụm cho dữ liệu mới; dùng random_state khi phù hợp. Lưu mô hình kèm thông số tiền xử lý cần thiết để sử dụng lại nếu phần việc yêu cầu.
7. **Evaluation & Visualization:** dùng Elbow và Silhouette để hỗ trợ chọn và đánh giá cụm; chỉ thêm chỉ số khác khi có lý do. Nêu công thức, cách đọc, hạn chế và điều kiện tính. Trực quan hóa cụm theo ba đặc trưng bằng biểu đồ 3D khi phù hợp; giải thích thang đo và tránh coi hình chiếu hai chiều là toàn bộ khoảng cách ba chiều.
8. **Conclusion:** mô tả đặc điểm từng nhóm bằng các giá trị đo được, đánh giá mức phù hợp và hạn chế; không suy diễn nguyên nhân hoặc hiệu quả kinh doanh từ tương quan hay phân cụm.

Không coi Elbow tự động xác định chính xác K; không so sánh các chỉ số ở những không gian hoặc tập dữ liệu khác nhau như thể tương đương. Không áp dụng Accuracy, Precision, Recall hay F1 như thể có nhãn thật khi không có nhãn.

## 4. Cấu trúc và mã nguồn

Điều chỉnh theo công việc thực tế, không tạo tệp rỗng chỉ để đủ cây thư mục. Ưu tiên `data/raw/`, `data/processed/`, `src/`, `notebooks/`, `models/`, `reports/`, `docs/images/`, `README.md`, `requirements.txt` và cấu hình nếu cần. Có thể dùng `results/` khi phù hợp, nhưng cần ghi rõ nơi lưu trong README.

- Gom các hàm liên quan để người học dễ theo dõi: đọc, kiểm tra, chia và chuẩn hóa dữ liệu có thể đặt chung trong một tệp tiền xử lý; tránh tồn tại hai tệp `preprocess` và `preprocessing` trùng vai trò.
- Đặt hàm vẽ dùng lại trong `src/visualize.py`; notebook import và gọi hàm, không sao chép cùng logic vẽ ở nhiều nơi. Tách mô hình và đánh giá thành các tệp rõ chức năng khi cần.
- Notebook dùng để giải thích, thực nghiệm và trình bày; `src/` chứa hàm tái sử dụng. Trước bước quan trọng, giải thích bằng Markdown: mục đích, vấn đề, đầu vào, nguyên lý/công thức, cách triển khai, đầu ra, diễn giải và cách kiểm chứng. Không cần lặp đầy đủ với đoạn code đơn giản.
- Tên hàm, biến và lớp bằng tiếng Anh. Mỗi hàm và lớp Python phải có docstring tiếng Việt ngay sau khai báo; giải thích mục đích và, khi phù hợp, đầu vào, đầu ra, giả định và lỗi. Chú thích gần các phép tính khó hiểu.
- Dùng thư viện thực sự cần thiết; không thêm design pattern hay tracking phức tạp. Không đổi tên, di chuyển hoặc xóa tệp ngoài phạm vi đã được yêu cầu.

## 5. Báo cáo và bằng chứng

- Khi bắt đầu triển khai lab, tạo báo cáo chính phù hợp, ví dụ `docs/Lab03_descript.md`: bài toán và dữ liệu; quy trình; cơ sở toán; triển khai; kết quả và biểu đồ; phân tích cụm; hạn chế và kết luận. Có thể bổ sung tài liệu giải thích tham số nếu cần.
- Làm đến bước nào, cập nhật báo cáo và trực quan hóa phù hợp với kết quả thực tế của bước đó ngay. Nếu chưa có dữ liệu, chưa chạy hoặc không có biểu đồ phù hợp, ghi rõ trạng thái và lý do; không tạo kết quả giả.
- Lưu kết quả thực nghiệm, cấu hình và biểu đồ tại thư mục phù hợp để đối chiếu; phân biệt rõ ví dụ minh họa và số liệu thực tế. Ghi seed, dữ liệu và các thiết lập cần tái lập.
- README nêu mục tiêu, nguồn dữ liệu khi xác minh được, cấu trúc, cách cài, chạy theo thứ tự và tìm báo cáo/kết quả. Nếu có demo thì hướng dẫn dùng; không tự thêm giao diện đồ họa khi chưa cần.
- Không giả định mọi nội dung của dự án mẫu đều bắt buộc: điều chỉnh biểu đồ, mô hình và chỉ số cho bài toán phân cụm. Không tự tạo nhật ký hay báo cáo tự động phức tạp.

## 6. Kiểm chứng và giải thích cho người học

- Chạy các notebook theo thứ tự từ trên xuống khi được tạo hoặc sửa; kiểm tra kích thước, kiểu, giá trị dữ liệu, tính tái lập và việc lưu/nạp mô hình nếu liên quan. Kiểm tra các trường hợp biên quan trọng của thuật toán.
- Phân biệt kết quả train, validation và test nếu sử dụng; không dùng test để chọn K hoặc điều chỉnh tham số. Nêu giới hạn khi tập đánh giá nhỏ hoặc một chỉ số không tính được.
- Giải thích bằng tiếng Việt, giữ thuật ngữ tiếng Anh và giải nghĩa lần đầu. Với khái niệm hoặc mã quan trọng, trình bày trực giác, mục đích, input/output, từng bước hoạt động, ví dụ thực tế, cách triển khai và cách kiểm tra; gắn với dữ liệu của bài khi phù hợp.
- Khi kết thúc công việc, nêu những gì đã làm, tệp quan trọng, cách kiểm tra và các vấn đề còn lại. Chỉ nêu lệnh đã kiểm tra hoặc ghi rõ nếu chưa chạy.
