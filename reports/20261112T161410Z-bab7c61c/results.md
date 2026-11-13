# Kết quả Lab 03

Lần chạy `20261112T161410Z-bab7c61c`.

## Dữ liệu

Có 200 dòng gốc và 200 dòng sau làm sạch. Số dòng bị loại do trùng là 0.

## Tìm cấu hình trên tập train

| family | k | silhouette_mean | silhouette_std | davies_bouldin_mean | calinski_harabasz_mean | valid_folds | converged_folds | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kmeans | 4 | 0.3778 | 0.0552 | 0.9517 | 19.0606 | 5 | 5 | True |
| kmeans | 6 | 0.3696 | 0.0744 | 0.8160 | 20.8339 | 5 | 5 | True |
| kmeans | 7 | 0.3641 | 0.0765 | 0.7896 | 19.5523 | 5 | 5 | True |
| kmeans | 5 | 0.3497 | 0.0601 | 0.9177 | 17.2764 | 5 | 5 | True |
| kmeans | 8 | 0.3424 | 0.1086 | 0.7330 | 19.2260 | 5 | 5 | True |
| gmm | 5 | 0.3372 | 0.0549 | 0.9464 | 16.6776 | 5 | 5 | True |
| kmeans | 9 | 0.3366 | 0.0831 | 0.7026 | 19.0761 | 5 | 5 | True |
| gmm | 6 | 0.3279 | 0.0893 | 0.9048 | 16.6333 | 5 | 5 | True |
| kmeans | 10 | 0.3259 | 0.0700 | 0.7192 | 19.1604 | 5 | 5 | True |
| kmeans | 2 | 0.3249 | 0.0238 | 1.2486 | 16.6883 | 5 | 5 | True |
| kmeans | 3 | 0.3235 | 0.0466 | 1.1575 | 14.6350 | 5 | 5 | True |
| gmm | 2 | 0.3150 | 0.0286 | 1.2513 | 16.5953 | 5 | 5 | True |
| gmm | 7 | 0.3104 | 0.0746 | 1.0065 | 15.8930 | 5 | 5 | True |
| gmm | 4 | 0.3049 | 0.0541 | 1.1213 | 13.7895 | 5 | 5 | True |
| gmm | 9 | 0.2977 | 0.0842 | 0.8770 | 15.6520 | 5 | 5 | True |
| gmm | 3 | 0.2767 | 0.0274 | 1.2486 | 12.6717 | 5 | 5 | True |
| gmm | 10 | 0.2326 | 0.1474 | 0.9330 | 14.1104 | 5 | 5 | True |
| gmm | 8 | 0.2292 | 0.1014 | 0.9169 | 11.8415 | 5 | 5 | True |

## Mô hình lưu

`kmeans` với `k=4`. Fit lại trên 160 khách và đánh giá trên 40 khách test.

## Chỉ số theo tập

| Tập | n_clusters_observed | smallest_cluster | silhouette | davies_bouldin | calinski_harabasz | negative_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| train | 4 | 27 | 0.4039 | 0.9412 | 98.7461 | 0.0063 |
| test | 4 | 5 | 0.3827 | 0.8718 | 26.4441 | 0.1000 |

## Độ ổn định khi lấy mẫu lại train

| repeats | mean_ari | std_ari | median_ari | min_ari | max_ari |
| --- | --- | --- | --- | --- | --- |
| 30 | 0.9950 | 0.0186 | 1.0000 | 0.9253 | 1.0000 |

## Hồ sơ cụm trên train

| cluster | count | share_pct | Age_mean | Annual Income (k$)_mean | Spending Score (1-100)_mean | silhouette_mean | negative_count | gender_Female | gender_Male |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 53 | 33.1250 | 53.9245 | 47.5094 | 39.9811 | 0.4027 | 0 | 32 | 21 |
| 1 | 27 | 16.8750 | 38.9259 | 87.9259 | 19.6296 | 0.2997 | 1 | 13 | 14 |
| 2 | 45 | 28.1250 | 25.7111 | 38.8889 | 61.2444 | 0.3550 | 0 | 27 | 18 |
| 3 | 35 | 21.8750 | 33.1429 | 87.2857 | 82.2571 | 0.5488 | 0 | 19 | 16 |

## Hồ sơ cụm trên test

| cluster | count | share_pct | Age_mean | Annual Income (k$)_mean | Spending Score (1-100)_mean | silhouette_mean | negative_count | gender_Female | gender_Male |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 12 | 30.0000 | 54.2500 | 48.5833 | 39.9167 | 0.2757 | 2 | 5 | 7 |
| 1 | 11 | 27.5000 | 40.4545 | 83.0000 | 19.4545 | 0.3937 | 2 | 6 | 5 |
| 2 | 12 | 30.0000 | 24.4167 | 44.1667 | 56.7500 | 0.3776 | 0 | 7 | 5 |
| 3 | 5 | 12.5000 | 31.0000 | 77.8000 | 76.4000 | 0.6274 | 0 | 3 | 2 |

Công thức, cách đọc và diễn giải lần chạy mặc định có trong [tài liệu Lab 03](../../docs/Lab_03_descript.md).
