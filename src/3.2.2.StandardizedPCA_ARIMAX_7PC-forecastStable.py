#3.2.2.StandardizedPCA_ARIMAX_7PC-forecastStable.py
# from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import skew, kurtosis
from tabulate import tabulate
import warnings
from google.colab import files  # Import files for download
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools

warnings.filterwarnings("ignore")

# 1.1 Đọc dữ liệu & Chuẩn hoá - step của phần trước
# Use a different variable name for the initial read to avoid confusion with the standardized df
df = pd.read_csv("VNINDEX_iqr_filled_HSG_Adjust.csv", index_col="Day", parse_dates=True)
# Chuẩn hóa z-score (bao gồm mean centering)
# Đọc dữ liệu đã chuẩn hóa
##===== Cải tiến công đoạn chuẩn hoá với StandardScaler ==============
# scaler = StandardScaler()
# df_standardized = pd.DataFrame(
#    scaler.fit_transform(df), columns=df.columns, index=df.index
# )
# print("Đã lưu dữ liệu chuẩn hóa vào VNINDEX_standardized.csv")
# 1.2 Kiểm tra trung bình và độ lệch chuẩn => Đã xử lý bước chuẩn hoá dữ liệu
# means = df_standardized.mean()
# stds = df_standardized.std()
## Đã có báo cáo riêng-chưa cần in
# print("Trung bình của dữ liệu đã chuẩn hóa:")
# print(means)
# print("\nĐộ lệch chuẩn của dữ liệu đã chuẩn hóa:")
# print(stds)
# df_standardized.to_csv('VNINDEX_standardized.csv')
# try:
#    files.download('VNINDEX_standardized.csv')
#    print("Đã lưu dữ liệu chuẩn hóa vào VNINDEX_standardized.csv")
# except NameError:
#    print("Chạy trong môi trường không phải Colab, bỏ qua download file.")
###------- thay thế bằng RobustScaler và xoá dữ liệu trùng ----
# Loại bỏ ngày trùng lặp (đề xuất thêm để cải thiện dữ liệu)
print(f"Số lượng ngày trước Orginal của dữ liệu: {len(df)}")
df = df.drop_duplicates()
print(f"Số lượng ngày sau khi loại bỏ trùng lặp: {len(df)}")
# Chuẩn hóa bằng RobustScaler
scaler = RobustScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
# 1.2 Kiểm tra trung vị và IQR
print("Trung vị trung bình của dữ liệu:", df_standardized.median().mean())
print("IQR trung bình của dữ liệu:", (df_standardized.quantile(0.75) - df_standardized.quantile(0.25)).mean())
means = df_standardized.mean()
stds = df_standardized.std()
## Đã có báo cáo riêng-chưa cần in
print("Trung bình của dữ liệu đã chuẩn hóa:")
print(means)
print("\nĐộ lệch chuẩn của dữ liệu đã chuẩn hóa:")
print(stds)

df_standardized.to_csv("VNINDEX_robust_standardized.csv")
try:
    files.download("VNINDEX_robust_standardized.csv")
    print("Đã lưu dữ liệu chuẩn hóa vào VNINDEX_robust_standardized.csv")
except NameError:
    print("Chạy trong môi trường không phải Colab, bỏ qua download file.")

print("Đã lưu dữ liệu chuẩn hóa vào VNINDEX_robust_standardized.csv")
##=============== END


# 2 Tính toán ma trận hiệp phương sai và ma trận tương quan Pearson
##2.1 Tách biến mục tiêu trước khi tính ma trận hiệp phương sai và tương quan
## Vai trò của VNINDEX trong phân tích & Lý do loại bỏ variant mục tiêu
##  VNINDEX là biến mục tiêu mà bạn muốn dự báo bằng mô hình ARIMAX. Trong bài toán này, mục tiêu là sử dụng các biến đầu vào (features) để giải thích và dự đoán giá trị của VNINDEX.
##  Khi xây dựng mô hình như ARIMAX, biến mục tiêu thường được tách ra khỏi tập dữ liệu đầu vào để tránh việc các biến độc lập bị ảnh hưởng bởi chính biến cần dự đoán, dẫn đến sai lệch trong phân tích.
##  Nếu bạn để VNINDEX trong X khi chạy PCA, nó sẽ bị coi là một biến đầu vào, điều này không đúng với mục tiêu phân tích (vì VNINDEX là biến cần dự đoán, không phải biến giải thích).
## Lợi ích loại bỏ VNIndex:
##  Tập trung vào biến độc lập: Bằng cách loại bỏ VNINDEX, ma trận hiệp phương sai và ma trận tương quan chỉ phản ánh mối quan hệ giữa các biến đầu vào, giúp PCA xác định các thành phần chính dựa trên sự biến thiên của các yếu tố độc lập.
##  Tránh rò rỉ dữ liệu (data leakage): Nếu VNINDEX được giữ lại trong quá trình PCA, thông tin từ biến mục tiêu có thể "rò rỉ" vào các thành phần chính, làm giảm tính khách quan và hiệu quả của mô hình ARIMAX khi dự báo.
##  Chuẩn bị cho ARIMAX: ARIMAX yêu cầu các biến ngoại sinh (exogenous variables) là các yếu tố độc lập với biến mục tiêu. Các thành phần chính từ PCA trên tập X (sau khi loại bỏ VNINDEX) sẽ đóng vai trò là biến ngoại sinh phù hợp.
if "VNINDEX" in df_standardized.columns:
    X_Variant = df_standardized.drop(columns=["VNINDEX"])
    y_Target = df_standardized["VNINDEX"]  # Keep target for later
else:
    print("Warning: 'VNINDEX' column not found in the standardized DataFrame.")
    raise ValueError("'VNINDEX' column not found after standardization. Cannot proceed.")

# 2.2 Tính Ma trận Hiệp phương sai từ dữ liệu đã chuẩn hóa
# Phương sai (variance) của một biến đã chuẩn hóa là 1
# Hiệp phương sai (covariance) giữa hai biến đã chuẩn hóa là tương quan của chúng
covariance_matrix = X_Variant.cov()

# 2.3 Tính Ma trận Tương quan (lý thuyết là giống ma trận hiệp phương sai của data chuẩn hóa)
correlation_matrix = X_Variant.corr()
## Tạm thời chưa cân in ma trân
##print("Ma trận hiệp phương sai (5x5 đầu tiên):")
##print(covariance_matrix.iloc[:5, :5])
##print("\nMa trận tương quan Pearson (5x5 đầu tiên):")
##print(correlation_matrix.iloc[:5, :5])
##print("\nMa trận tương quan:\n", correlation_matrix)

# So sánh hai ma trận bằng np.allclose
# allclose kiểm tra xem hai mảng có gần giống nhau trong dung sai cho phép hay không
# rtol (relative tolerance): dung sai tương đối
# atol (absolute tolerance): dung sai tuyệt đối
are_matrices_close = np.allclose(covariance_matrix, correlation_matrix, atol=1e-5)

print(
    f"Ma trận Hiệp phương sai và Ma trận Tương quan có gần giống nhau không (sử dụng np.allclose)? {are_matrices_close}"
)

# . Lưu dữ liệu chuẩn hóa và ma trận tương quan
correlation_matrix.to_csv("VNINDEX_correlation_matrix.csv")
covariance_matrix.to_csv("VNINDEX_covariance_matrix.csv")
try:
    files.download("VNINDEX_correlation_matrix.csv")
    files.download("VNINDEX_covariance_matrix.csv")
    print("\nĐã lưu ma trận tương quan vào VNINDEX_correlation_matrix.csv")
    print("Đã lưu ma trận hiệp phương sai VNINDEX_covariance_matrix.csv")
except NameError:
    print("\nRunning in a non-Colab environment, skipping file downloads.")
    print("Files saved locally: VNINDEX_correlation_matrix.csv, VNINDEX_covariance_matrix.csv")

print("\nMa trận Tương quan Pearson:\n")
# Sử dụng tabulate để hiển thị ma trận đẹp hơn trong output console
# Đã đạt - hạn chế print Console
# from tabulate import tabulate
# print(tabulate(correlation_matrix, headers='keys', tablefmt='grid', floatfmt=".4f"))

# 2.4 Trực quan hóa Ma trận Tương quan bằng Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Ma trận Tương quan Pearson (Không bao gồm VNINDEX)")
plt.tight_layout()
plt.savefig("Correlation_heatmap.png")
plt.show()
plt.close()
try:
    files.download("Correlation_heatmap.png")
    print("\nĐã lưu heatmap ma trận tương quan vào 'correlation_heatmap.png'")
except NameError:
    print("\nRunning in a non-Colab environment, skipping file downloads.")
    print("Heatmap saved locally: Correlation_heatmap.png")


# 2.5. Đánh giá Mối quan hệ Tương quan (trên X_Variant)
# print("\n--- Đánh giá Chi tiết Mối quan hệ Tương quan ---")

# 2.5.1 Kiểm tra/tìm các cặp biến có tương quan cao (ví dụ: > 0.8 hoặc < -0.8, loại bỏ tương quan với chính nó)
# Xem xet có cần đưa vào chương trình?
high_corr_threshold = 0.8
high_corr_pairs = {}
totalVariant = 0
# Lặp qua ma trận tương quan chỉ ở tam giác dưới để tránh lặp và tương quan chính nó
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]
        totalVariant = totalVariant + 1
        if abs(corr_value) >= high_corr_threshold:
            high_corr_pairs[(col1, col2)] = corr_value

if high_corr_pairs:
    print(
        f"\n Tổng số cặp biến :{len(high_corr_pairs)} có |r| >= {high_corr_threshold} trên tổng số cặp biến : {totalVariant}"
    )
    print(f"\nCác cặp biến có tương quan |r| >= {high_corr_threshold}:")
    for pair, corr in high_corr_pairs.items():
        print(f"  {pair[0]} và {pair[1]}: {corr:.4f}")
    print("\nNhận xét: Các cặp biến này có mối quan hệ tuyến tính mạnh, có thể chứa thông tin trùng lặp.")
    print("PCA sẽ hiệu quả trong việc kết hợp các biến này thành ít thành phần hơn.")
else:
    print(f"\nKhông tìm thấy cặp biến nào có tương quan |r| >= {high_corr_threshold}.")
    print(
        "Nhận xét: Các biến có vẻ tương đối độc lập theo nghĩa tuyến tính. PCA vẫn có thể giúp tìm các thành phần chính, nhưng mức độ giảm chiều có thể không lớn nếu không có nhóm biến tương quan cao."
    )

# 2.5.2 Kiểm tra/Tìm các cặp biến có tương quan thấp (ví dụ: |r| < 0.3)
low_corr_threshold = 0.3
low_corr_pairs = {}
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]

        if abs(corr_value) < low_corr_threshold:
            low_corr_pairs[(col1, col2)] = corr_value

# Chỉ in một vài cặp tiêu biểu nếu số lượng quá lớn
num_low_corr_to_print = 10
if low_corr_pairs:
    print(f"\nMột vài cặp biến có tương quan |r| < {low_corr_threshold} (tiêu biểu {num_low_corr_to_print} cặp):")
    for i, (pair, corr) in enumerate(low_corr_pairs.items()):
        if i >= num_low_corr_to_print:
            break
        print(f"  {pair[0]} và {pair[1]}: {corr:.4f}")
    print(
        "\nNhận xét: Các cặp biến này có mối quan hệ tuyến tính yếu. Chúng có thể đại diện cho các yếu tố độc lập trong dữ liệu."
    )
else:
    print(f"\nKhông tìm thấy cặp biến nào có tương quan |r| < {low_corr_threshold}.")

# 2.5.4 Kiểm tra phương sai để xác định ngưỡng (trên X_Variant)
# Ngưỡng nên được điều chỉnh dựa trên phân phối phương sai của các biến trong covariance_matrix.
# Kiểm tra phân phối này bằng cách in ra variances = np.diag(correlation_matrix) và chọn ngưỡng dựa trên phân vị
# (ví dụ: 5th percentile hoặc 10th percentile) có thể thay đổi
variances = np.diag(covariance_matrix)
percentile_threshold = 5
# Set low variance threshold - adjust based on percentile if needed

low_variance_threshold = np.percentile(variances, percentile_threshold)
print(f"\nPhương sai nhỏ nhất (trên X_Variant): {variances.min():.4f}")
print(f"Phương sai trung bình (trên X_Variant): {variances.mean():.4f}")
print(
    f"Phân vị {percentile_threshold}% của variances (trên X_Variant): {np.percentile(variances, percentile_threshold):.4f}"
)
print(f"Ngưỡng phương sai thấp được sử dụng (dựa trên {percentile_threshold}% phân vị): {low_variance_threshold:.4f}")


# 3. Dùng pca.fit(X_Variant) để phân tích ban đầu - Chỉ huấn luyện mô hình
pca = PCA()
if not X_Variant.empty:
    pca.fit(X_Variant)

    # 3.1 Tính eigenvalues (phương sai của từng thành phần chính)
    eigenvalues = pca.explained_variance_
    print("\nEigenvalues (phương sai của từng PC):", eigenvalues[:5])
    print(
        "Ý nghĩa: Eigenvalues thể hiện phương sai mà mỗi thành phần chính giải thích. Giá trị càng lớn, PC càng quan trọng."
    )

    # 3.2 Tính eigenvectors (các vector riêng, định hướng của PC)
    eigenvectors = pca.components_
    print("\nEigenvectors (hướng của PC1):", eigenvectors[0][:5])
    print("Ý nghĩa: Eigenvectors định hướng cho mỗi PC, thể hiện mức độ đóng góp của từng biến gốc vào PC.")

    # 3.3 Tính tỷ lệ phương sai giải thích
    ##Là mức độ phương sai trong dữ liệu gốc mà mỗi thành phần chính (PC) giải thích được.
    explained_variance_ratio = pca.explained_variance_ratio_
    print("\nTỷ lệ phương sai giải thích dem lại gia trị này:", explained_variance_ratio[:5])

    # 3.4 Tính tỷ lệ phương sai tích lũy
    ##Là tổng cộng dồn của các phương sai giải thích từ PC₁ đến PCₙ. Nó cho biết tổng lượng thông tin (phương sai) được giữ lại khi chọn 𝑛 thành phần chính đầu tiên.
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("\n--- Phân tích PCA ---")
    print("Tỷ lệ phương sai giải thích bởi từng thành phần chính:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"  PC{i+1}: {ratio:.4f} ({cumulative_variance[i]:.4f} tích lũy)")

    print("Tỷ lệ phương sai tích lũy:", cumulative_variance[: min(5, len(cumulative_variance))])
    print(
        "Ý nghĩa: Tỷ lệ phương sai tích lũy cho biết tổng phương sai được giải thích bởi các PC. Thường cần >= 80% để đảm bảo dữ liệu giảm chiều tốt."
    )
    # ĐÁNH GIÁ METRIC NAY CÓ TRÙNG LĂP - START
    # 3.5.1 Đánh giá dựa trên phương sai giải thích
    # Ngưỡng phương sai tích lũy variance_threshold (cumulative_variance_threshold) > 90%
    # Chuẩn phổ biến khi áp dụng cho PCA + ARIMA / ARIMAX	Thỏa mãn độ chính xác & tránh overfitting với Kỳ vọng Số lượng PC điển hình từ 5~7
    variance_threshold = 0.95
    n_components_to_keep = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"\n--- Đánh giá PCA ---")
    print(f"Tổng số biến gốc (sau tách VNINDEX): {X_Variant.shape[1]}")
    print(f"Ngưỡng phương sai tích lũy mong muốn: {variance_threshold*100:.0f}%")
    print(f"Số thành phần chính PCA có thể giữ lại: {n_components_to_keep}")
    print(f"\n--- Đánh giá PCA tiêu chí 1 ---")
    # 3.5.1 Đảm bảo ngưỡng có thể đạt được
    if n_components_to_keep <= X_Variant.shape[1]:
        print(
            f"Để giải thích ít nhất {variance_threshold*100:.0f}% phương sai, cần {n_components_to_keep} thành phần chính."
        )
        reduction_percentage = (1 - n_components_to_keep / X_Variant.shape[1]) * 100
        print(
            f"Có thể giảm chiều dữ liệu từ {X_Variant.shape[1]} xuống {n_components_to_keep} chiều (giảm khoảng {reduction_percentage:.2f}%)."
        )
        print(
            "\nNhận xét: PCA cho phép giảm đáng kể số chiều dữ liệu trong khi vẫn giữ lại phần lớn thông tin (phương sai). Các thành phần chính này sẽ là các đặc trưng mới cho các mô hình tiếp theo."
        )
    else:
        print(f"Ngưỡng phương sai {variance_threshold*100:.0f}% không thể đạt được ngay cả với tất cả các thành phần.")
        print(f"Tổng phương sai giải thích tối đa là {cumulative_variance[-1]*100:.2f}%.")
        print(
            "Nhận xét: Việc giảm chiều bằng PCA không hiệu quả nhiều trong trường hợp này nếu mục tiêu là giữ lại tỷ lệ phương sai rất cao."
        )

    print(f"\n--- Đánh giá PCA tiêu chí 2 ---")

    if cumulative_variance[-1] >= variance_threshold:
        print(
            f"Để giải thích ít nhất {variance_threshold*100:.0f}% phương sai, cần {n_components_to_keep} thành phần chính."
        )
        reduction_percentage = (1 - n_components_to_keep / X_Variant.shape[1]) * 100
        print(
            f"Có thể giảm chiều dữ liệu từ {X_Variant.shape[1]} xuống {n_components_to_keep} chiều (giảm khoảng {reduction_percentage:.2f}%)."
        )
        print(
            "\nNhận xét: PCA cho phép giảm đáng kể số chiều dữ liệu trong khi vẫn giữ lại phần lớn thông tin (phương sai). Các thành phần chính này sẽ là các đặc trưng mới cho các mô hình tiếp theo."
        )
    else:
        print(
            f"Ngay cả khi giữ lại tất cả {X_Variant.shape[1]} thành phần, tổng phương sai giải thích là {cumulative_variance[-1]*100:.2f}%."
        )
        print("Nhận xét: Có thể tập dữ liệu này không chứa nhiều thông tin dư thừa (biến tương quan cao);")
        print(
            "          hoặc phương sai phân bổ đều trên nhiều chiều. Việc giảm chiều bằng PCA có thể không hiệu quả nhiều trong trường hợp này nếu mục tiêu là giữ lại tỷ lệ phương sai rất cao."
        )

    # 3.5.2 Đánh giá: 5 thành phần chính có đủ tốt không?
    print(f"\n--- Đánh khả năng giảm chiều các thành phần chính PCA. : ")
    if len(cumulative_variance) > 4:  # Check if 5th PC (index 4) exists
        if cumulative_variance[4] >= variance_threshold:
            print(
                f"{n_components_to_keep}  thành phần chính giải thích {cumulative_variance[4]*100:.2f}% phương sai, đủ tốt để giảm chiều cho ARIMAX."
            )
        else:
            print(
                f"{n_components_to_keep} thành phần chính chỉ giải thích {cumulative_variance[4]*100:.2f}% phương sai, có thể cần thêm PC."
            )
    else:
        print(
            f"Chỉ có {len(cumulative_variance)} thành phần chính. Tổng phương sai giải thích là {cumulative_variance[-1]*100:.2f}%."
        )

    print(f"\n--- Kết luận khả năng giảm chiều PCA ---")
    print(f"Tổng số biến gốc: {df_standardized.shape[1]}")  # Includes VNINDEX here
    print(f"Ngưỡng phương sai tích lũy mong muốn: {variance_threshold*100:.0f}%")
    if len(cumulative_variance) > 4:
        print(f"Ngưỡng phương sai tích lũy đạt được (với 5 PC): {cumulative_variance[4]*100:.0f}%")
    else:
        print("Không đủ 5 thành phần chính.")

    print(f"Số thành phần chính PCA có thể giữ lại theo ngưỡng {variance_threshold*100:.0f}%: {n_components_to_keep}")
    # -- END
    # 3.6 Biểu đồ tỷ lệ phương sai tích lũy
    # Trực quan hóa Tỷ lệ Phương sai Giải thích
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(explained_variance_ratio) + 1),
        cumulative_variance,
        marker="o",
        linestyle="--",
    )
    plt.xlabel("Số lượng thành phần chính")
    plt.ylabel("Tổng Phương sai Giải thích (%)")
    plt.title("Biểu đồ Phương sai Giải thích Tích lũy của PCA")
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(True)
    plt.savefig("pca_explained_variance.png")
    plt.show()
    plt.close()
    try:
        files.download("pca_explained_variance.png")
        print("\nĐã lưu biểu đồ phương sai giải thích tích lũy vào 'pca_explained_variance.png'")
    except NameError:
        print("\nRunning in a non-Colab environment, skipping file downloads.")
        print("Plot saved locally: pca_explained_variance.png")

    # 4. Tính toán và lọc dữ liệu dựa trên ma trận hiệp phương sai, ma trận tương quan và loadings
    # Tính vector tương quan (loadings)    # Tính vector tương quan (loadings) trên X_Variant
    if pca.n_components_ > 0:
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings, columns=[f"PC{i+1}" for i in range(loadings.shape[1])], index=X_Variant.columns
        )
        print("\nLoadings cho PC1 (đã được sắp xếp giảm dần):")
        if "PC1" in loadings_df.columns:
            print(loadings_df["PC1"].sort_values(ascending=False).head(min(5, len(loadings_df["PC1"]))))
            print("Ý nghĩa: Loadings thể hiện mức độ đóng góp của các biến gốc vào PC1.")
        else:
            print("PC1 does not exist in loadings_df. Cannot print loadings.")
    else:
        print("\nPCA resulted in 0 components on X_Variant. Cannot calculate loadings.")

    # 4.1 Loại nhiễu: Dựa trên ma trận hiệp phương sai (phương sai thấp) và loadings (đóng góp thấp)
    # Identify low variance variables based on the calculated threshold
    low_variance_vars = covariance_matrix.columns[variances < low_variance_threshold].tolist()
    print(f"\nBiến có phương sai thấp (dựa trên ngưỡng {low_variance_threshold:.4f}): {low_variance_vars}")
    ##Ngưỡng trung bình tuyệt đối của loadings (vector tương quan) được sử dụng để xác định các biến có đóng góp thấp vào các thành phần chính (PCs) và có thể bị coi là nhiễu.
    ##Giá trị này đại diện cho mức độ quan trọng trung bình của một biến trong việc giải thích phương sai của dữ liệu qua tất cả các PCs.
    ##Nếu trung bình tuyệt đối của loadings của một biến nhỏ hơn 0.1, biến đó được cho là không đủ ý nghĩa và có thể bị loại bỏ.
    ##Ngưỡng low_contribution_threshold = 0.1 là hợp lý cho dữ liệu chuẩn hóa, có thể kiểm tra phân phối mean_loadings để điều chỉnh (tăng lên 0.2 hoặc giảm xuống 0.05 nếu cần).

    # Identify low contribution variables based on loadings (on X_Variant)
    # Determine a threshold based on the distribution of mean loadings
    if "loadings_df" in locals() and not loadings_df.empty:
        mean_loadings = np.abs(loadings_df).mean(axis=1)
        # Example: Keep top 80% variables by mean loading
        # Keep variables above the 20th => 30 percentile of mean loadings
        # Tăng ngưỡng đóng góp thấp từ phân vị 20% lên 30% để giữ lại nhiều biến hơn, tránh loại bỏ các biến quan trọng
        percentile_loading_threshold = 30
        low_contribution_threshold = np.percentile(mean_loadings, percentile_loading_threshold)
        print(
            f"Ngưỡng đóng góp thấp được sử dụng (dựa trên {percentile_loading_threshold}th percentile của mean loadings): {low_contribution_threshold:.4f}"
        )

        low_contribution_vars = mean_loadings[mean_loadings < low_contribution_threshold].index.tolist()
        print(f"Số biến có mean_loadings < {low_contribution_threshold:.4f}: {len(low_contribution_vars)}")
        print(f"Danh sách mean_loadings thấp nhất:\n{mean_loadings.sort_values().head(10)}")  # Print more if needed
    else:
        print("loadings_df is not available or is empty. Skipping low contribution variable check.")
        low_contribution_vars = []  # Ensure this list is empty if loadings_df is not available

    # Combine low variance and low contribution variables to identify noise
    noisy_variables = list(set(low_variance_vars + low_contribution_vars))
    print(f"\nBiến nhiễu bị loại bỏ: {noisy_variables}")

    # Create df_cleaned_noise by dropping noisy variables from df_standardized
    df_cleaned_noise = df_standardized.drop(columns=noisy_variables)
    print(f"Số lượng biến sau khi lọc nhiễu: {df_cleaned_noise.shape[1]}")  # Includes VNINDEX potentially

    # 4.2 Loại bỏ biến dư thừa: Dựa trên ma trận tương quan Pearson (tương quan > 0.8) on df_cleaned_noise
    variables_to_drop_redundant = set()
    # Only proceed if there are enough columns for correlation matrix after noise cleaning
    if not df_cleaned_noise.empty and df_cleaned_noise.shape[1] > 1:
        # Ensure 'VNINDEX' is excluded from redundant variable check
        df_cleaned_noise_features = (
            df_cleaned_noise.drop(columns=["VNINDEX"])
            if "VNINDEX" in df_cleaned_noise.columns
            else df_cleaned_noise.copy()
        )

        if not df_cleaned_noise_features.empty and df_cleaned_noise_features.shape[1] > 1:
            corr_matrix_cleaned = df_cleaned_noise_features.corr()
            highly_correlated_pairs_cleaned = []
            for i in range(len(corr_matrix_cleaned.columns)):
                for j in range(i + 1, len(corr_matrix_cleaned.columns)):
                    if (
                        abs(corr_matrix_cleaned.iloc[i, j]) > high_corr_threshold
                    ):  # Use the high_corr_threshold defined earlier
                        highly_correlated_pairs_cleaned.append(
                            (
                                corr_matrix_cleaned.columns[i],
                                corr_matrix_cleaned.columns[j],
                                corr_matrix_cleaned.iloc[i, j],
                            )
                        )

            for var1, var2, corr in highly_correlated_pairs_cleaned:
                # Decide which one to drop - here we drop var2 from the redundant pair
                # Ensure the variable is not the target variable
                if var1 != "VNINDEX" and var2 != "VNINDEX":
                    # To avoid adding the same variable multiple times if it's correlated with several others
                    if var2 not in variables_to_drop_redundant:
                        variables_to_drop_redundant.add(var2)

            df_cleaned = df_cleaned_noise.drop(columns=list(variables_to_drop_redundant))
            print(
                f"\nBiến dư thừa bị loại bỏ (dựa trên ma trận Pearson trên dữ liệu đã lọc nhiễu): {list(variables_to_drop_redundant)}"
            )
        else:
            print("\nKhông đủ biến (sau lọc nhiễu và loại VNINDEX) để kiểm tra biến dư thừa.")
            df_cleaned = df_cleaned_noise.copy()  # Keep the same DataFrame
    else:
        print("\ndf_cleaned_noise is empty or has only one column. Skipping redundant variable check.")
        df_cleaned = df_cleaned_noise.copy()  # Keep the same DataFrame

else:  # Case where initial X_Variant was empty
    print("Initial X_Variant DataFrame is empty. Cannot proceed with PCA analysis and filtering.")
    # Set df_cleaned to an empty state or handle appropriately
    df_cleaned = pd.DataFrame(index=df_standardized.index)
    explained_variance_ratio = np.array([])
    cumulative_variance = np.array([])
    n_components_to_keep = 0
    # Skip subsequent steps that depend on PCA and filtering


# 5.Pha giảm chiều  Áp dụng PCA trên dữ liệu đã lọc và loại bỏ biến mục tiêu VNINDEX
# Ensure df_cleaned has columns and contains VNINDEX before proceeding
X_cleaned = pd.DataFrame()
y = pd.Series()
X_pca_df = pd.DataFrame()  # Initialize X_pca_df

if "VNINDEX" in df_cleaned.columns and df_cleaned.shape[1] > 1:  # Need VNINDEX and at least one feature for PCA
    X_cleaned = df_cleaned.drop(columns=["VNINDEX"])
    y = df_cleaned["VNINDEX"]
    print(f"\nShape of X_cleaned before final PCA: {X_cleaned.shape}")

    # Check if X_cleaned is empty before applying final PCA
    if not X_cleaned.empty and X_cleaned.shape[1] > 0:
        # 5.1 Verify các Số thành phần chính
        # Ensure n_components_to_keep is valid for X_cleaned dimensions
        # Decide on the number of components for the final PCA
        # Option 1: Use n_components_to_keep from the initial PCA based on the cumulative variance threshold
        # Option 2: Choose a fixed number, e.g., 5, if that's the project requirement
        # Let's use 5 as the user's last code snippet intended, but ensure it doesn't exceed the number of features
        # Use 5 PCs, but no more than the available features
        # variance_threshold = 0.95 => Có 7 PC sử dụng giá trị suy luận từ variance_threshold là n_components_to_keep để tính toán
        n_components_final_pca = min(n_components_to_keep, X_cleaned.shape[1])

        if n_components_final_pca > 0:
            print(f"Applying final PCA with n_components = {n_components_final_pca}")
            pca_final = PCA(n_components=n_components_final_pca)
            X_pca_final = pca_final.fit_transform(X_cleaned)

            # Tạo DataFrame cho các thành phần chính
            X_pca_df = pd.DataFrame(
                X_pca_final, columns=[f"PC{i+1}" for i in range(n_components_final_pca)], index=df_cleaned.index
            )

            # Tính lại tỷ lệ phương sai tích lũy sau khi lọc
            explained_variance_ratio_final = pca_final.explained_variance_ratio_ * 100
            cumulative_variance_final = np.cumsum(explained_variance_ratio_final)

            # Tính loadings sau khi lọc
            loadings_final = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)
            loadings_final_df = pd.DataFrame(
                loadings_final, columns=[f"PC{i+1}" for i in range(n_components_final_pca)], index=X_cleaned.columns
            )
            ###===== Bổ sung
            ### 5.1.1 vẽ biểu đồ histogram, boxplot, hoặc kiểm tra skewness/kurtosis để bổ sung phân tích thăm dò.

            # Thiết lập kiểu dáng cho biểu đồ với hình nền trắng
            # Sử dụng style 'white' hoặc 'whitegrid'
            #sns.set_style('whitegrid')  # Set the style to 'white' for a plain white background
            # If you still want subtle grid lines, use 'whitegrid' instead:
            sns.set_style('whitegrid')
            sns.set_context('notebook')
            sns.set(font_scale=1.2)

            # The rest of your code remains the same:
            # Đọc dữ liệu đã chuẩn hóa
            #df_standardized = pd.read_csv("VNINDEX_robust_standardized.csv", index_col="Day", parse_dates=True)

            # Danh sách biến ngoại sinh sau khi lọc (df_cleaned)
            # Make sure 'VNINDEX' is included if you plan to plot it as the target
            selected_vars = df_cleaned

            # Lọc dữ liệu để chỉ lấy các biến cần phân tích
            # Ensure selected_vars contains only columns present in df_standardized
            actual_selected_vars = [col for col in selected_vars if col in df_standardized.columns]
            df_eda = df_standardized[actual_selected_vars]

            if df_eda.empty:
                print("Warning: The filtered DataFrame for EDA is empty. Check 'selected_vars' and 'df_standardized.columns'.")
            else:
                # 1. Tính toán skewness và kurtosis
                stats_table = []
                for col in df_eda.columns:
                    # Ensure there are non-NaN values to compute stats
                    if df_eda[col].dropna().shape[0] > 1:
                        skewness = skew(df_eda[col].dropna())
                        kurt = kurtosis(df_eda[col].dropna(), fisher=True)  # Fisher=True để kurtosis chuẩn hóa (0 cho phân phối chuẩn)
                        stats_table.append([col, f"{skewness:.4f}", f"{kurt:.4f}"])
                    else:
                        stats_table.append([col, "N/A", "N/A"]) # Handle columns with all NaNs

                print("\nThống kê Skewness và Kurtosis của các biến:")
                print(tabulate(stats_table, headers=["Biến", "Skewness", "Kurtosis"], tablefmt="grid", numalign="right"))

                # 2. Vẽ biểu đồ Histogram
                # Adjust grid size based on the number of selected variables
                n_vars = len(df_eda.columns)
                n_cols = 3
                n_rows = (n_vars + n_cols - 1) // n_cols # Calculate rows needed

                plt.figure(figsize=(n_cols * 5, n_rows * 4)) # Adjust figure size dynamically
                for i, col in enumerate(df_eda.columns, 1):
                    plt.subplot(n_rows, n_cols, i)
                    # Only plot if there's data
                    if df_eda[col].dropna().shape[0] > 0:
                        sns.histplot(df_eda[col].dropna(), bins=30, kde=True, color='blue')
                        plt.title(f'Histogram của {col}')
                        plt.xlabel(col)
                        plt.ylabel('Tần số')
                    else:
                        plt.title(f'Không có dữ liệu để vẽ Histogram cho {col}')
                        # Optionally plot an empty subplot or text
                        plt.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                plt.savefig("eda_histogram.png", dpi=300)
                plt.show()
                plt.close()

                try:
                    if files:
                        files.download("eda_histogram.png")
                        print("Đã lưu biểu đồ histogram vào 'eda_histogram.png'")
                except NameError:
                    print("Chạy trong môi trường không phải Colab, bỏ qua tải file.")
                    print("Biểu đồ đã được lưu cục bộ: eda_histogram.png")

                # 3. Vẽ biểu đồ Boxplot
                plt.figure(figsize=(15, 8))
                # Ensure the DataFrame for boxplot is not empty
                if not df_eda.empty:
                    sns.boxplot(data=df_eda, palette="Set2")
                    plt.title("Boxplot của các biến (chuẩn hóa)")
                    plt.xticks(rotation=45)
                    plt.ylabel("Giá trị chuẩn hóa")
                    plt.savefig("eda_boxplot.png", dpi=300)
                    plt.show()
                    plt.close()

                    try:
                        if files:
                            files.download("eda_boxplot.png")
                            print("Đã lưu biểu đồ boxplot vào 'eda_boxplot.png'")
                    except NameError:
                        print("Chạy trong môi trường không phải Colab, bỏ qua tải file.")
                        print("Biểu đồ đã được lưu cục bộ: eda_boxplot.png")
                else:
                    print("\nKhông có dữ liệu để vẽ Boxplot.")


                # 4. Vẽ biểu đồ chuỗi thời gian cho VNINDEX
                # Check if VNINDEX column exists in the filtered df_eda
                if 'VNINDEX' in df_eda.columns and not df_eda['VNINDEX'].empty:
                    plt.figure(figsize=(12, 6))
                    plt.plot(df_eda.index, df_eda['VNINDEX'], label='VNINDEX (chuẩn hóa)', color='blue')
                    plt.title("Chuỗi thời gian VNINDEX (chuẩn hóa)")
                    plt.xlabel("Ngày")
                    plt.ylabel("Giá trị chuẩn hóa")
                    plt.legend()
                    plt.grid(True)
                    plt.xticks(rotation=45)
                    plt.savefig("eda_timeseries_vnindex.png", dpi=300)
                    plt.show()
                    plt.close()

                    try:
                        if files:
                            files.download("eda_timeseries_vnindex.png")
                            print("Đã lưu biểu đồ chuỗi thời gian vào 'eda_timeseries_vnindex.png'")
                    except NameError:
                        print("Chạy trong môi trường không phải Colab, bỏ qua tải file.")
                        print("Biểu đồ đã được lưu cục bộ: eda_timeseries_vnindex.png")
                else:
                    print("\nKhông tìm thấy cột 'VNINDEX' trong dữ liệu đã lọc hoặc dữ liệu trống. Không thể vẽ biểu đồ chuỗi thời gian VNINDEX.")

            # 5. In bảng eigenvalues chi tiết
            eigenvalues_table = []
            for i in range(7):
                eigenvalues_table.append([f'PC{i+1}', f'{eigenvalues[i]:.4f}', f'{explained_variance_ratio_final[i]:.2f}%', f'{cumulative_variance_final[i]:.2f}%'])
            print("\nBảng Eigenvalues và Tỷ lệ Phương sai:")
            print(tabulate(eigenvalues_table, headers=["Thành phần", "Eigenvalue", "Tỷ lệ (%)", "Tích lũy (%)"], tablefmt="grid", numalign="right"))

            # 2. Vẽ biểu đồ loadings của 7 PC
            plt.figure(figsize=(15, 10))
            for i in range(7):
                plt.subplot(4, 2, i+1)  # Sắp xếp trong lưới 4x2
                sns.barplot(x=loadings_final_df.index, y=loadings_final_df[f'PC{i+1}'], palette="viridis")
                plt.title(f'Loadings của PC{i+1}')
                plt.xlabel('Biến')
                plt.ylabel('Hệ số Loadings')
                plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("loadings_final.png", dpi=300)
            plt.show()
            plt.close()
            try:
                if files:
                    files.download("loadings_final.png")
                    print("Đã lưu biểu đồ loadings vào 'loadings_final.png'")
            except NameError:
                print("Chạy trong môi trường không phải Colab, bỏ qua tải file.")
                print("Biểu đồ đã được lưu cục bộ: loadings_final.png")

        # 5.1.2 - CHeck In bảng kết quả PC1~PC(n) với top 5/7 biến và p-value ADF
        # for i in range(n_components_final_pca):
        #  print(f"PC{i+1}: {explained_variance_ratio_final[i]:.2f}% ({cumulative_variance_final[i]:.2f}% tích lũy)")
        # --- thay thế bằng bảng chi tiết
        table_data = []
        for i in range(n_components_final_pca):
            # Tính top 5 biến đóng góp
            abs_loadings = np.abs(loadings_final_df[f"PC{i+1}"])
            total_abs_loadings = abs_loadings.sum()
            #top_pc_vars = abs_loadings.sort_values(ascending=False).head(n_components_final_pca).index
            #top_pc_contrib = [
            #    f"{var} ({(abs_loadings[var] / total_abs_loadings * 100):.2f}%)" for var in top_pc_vars
            #]
            #top_pc_str = ", ".join(top_pc_contrib)

            # Kiểm tra ADF cho PC
            adf_result = adfuller(X_pca_df[f"PC{i+1}"])
            p_value = adf_result[1]

            table_data.append(
                [
                    f"PC{i+1}",
                    f"{explained_variance_ratio_final[i]:.2f}%",
                    f"{cumulative_variance_final[i]:.2f}%",
                    #top_pc_str,
                    f"{p_value:.4f}",
                ]
            )

        print(f"\nBảng thông tin các thành phần chính (PC1~PC{n_components_final_pca}):")
        print(
            tabulate(
                table_data,
                headers=[
                    "Thành phần chính",
                    "Phương sai giải thích (%)",
                    "Phương sai tích lũy (%)",
                    #f"Top {n_components_final_pca} biến đóng góp (% đóng góp)",
                    "p-value ADF",
                ],
                numalign="right",
                stralign="center",
                maxcolwidths=[20, 20, 20, 100],
                tablefmt="grid",
            )
        )

        if cumulative_variance_final[-1] >= 80:
            # Check the last cumulative variance against a reasonable threshold
            print(
                f"Ý nghĩa Tỷ lệ phương sai tích lũy {cumulative_variance_final[-1]:.2f}% trên dữ liệu đã lọc và giảm chiều là phù hợp để sử dụng trong ARIMAX."
            )
        else:
            print(
                f"Ý nghĩa Tỷ lệ phương sai tích lũy {cumulative_variance_final[-1]:.2f}% sau lọc/giảm chiều thấp hơn 80%."
            )

        # 5.3Kiểm tra n_components_final_pca thành phần chính giải thích có đủ tốt
        print(
            f"\nKiểm tra số thành phân chính cho giảm chiều tập train {n_components_final_pca} thành phần chính giải thích có đủ tốt"
        )
        if len(cumulative_variance) > n_components_to_keep - 1:
            # Check if 7th PC (index 6) exists
            if cumulative_variance[n_components_final_pca - 1] >= variance_threshold:
                print(
                    f"{n_components_final_pca} thành phần chính giải thích {cumulative_variance[n_components_final_pca-1]*100:.2f}% phương sai, đủ tốt để giảm chiều cho ARIMAX."
                )
            else:
                print(
                    f"{n_components_final_pca} thành phần chính chỉ giải thích {cumulative_variance[n_components_final_pca-1]*100:.2f}% phương sai, có thể cần thêm PC."
                )
        else:
            print(
                f"Chỉ có {len(cumulative_variance)} thành phần chính. Tổng phương sai giải thích là {cumulative_variance[-1]*100:.2f}%."
            )
        #### ======= END BỔ SUNG KIỂM TRA & BIỂU ĐỒ 5.1
        ##===== Optimize Start===
        ##5.2. Kiểm tra overfitting của PCA (Thay vì #8 Kiểm tra ở cuối cùng
        train_size = int(len(X_cleaned) * 0.8)
        X_train_cleaned = X_cleaned[:train_size]
        X_test_cleaned = X_cleaned[train_size:]
        pca_train_final = PCA(n_components=n_components_final_pca)
        X_train_pca_final = pca_train_final.fit_transform(X_train_cleaned)
        X_test_pca_final = pca_train_final.transform(X_test_cleaned)

        X_train_reconstructed_final = pca_train_final.inverse_transform(X_train_pca_final)
        X_test_reconstructed_final = pca_train_final.inverse_transform(X_test_pca_final)

        mse_train_final = mean_squared_error(X_train_cleaned, X_train_reconstructed_final)
        mse_test_final = mean_squared_error(X_test_cleaned, X_test_reconstructed_final)
        print(f"\nMSE trên tập train (PCA): {mse_train_final:.4f}")
        print(f"MSE trên tập test (PCA): {mse_test_final:.4f}")
        print(f"Chênh lệch MSE (PCA, càng nhỏ càng tốt): {abs(mse_train_final - mse_test_final):.4f}")
        ##===== Optimize end===

        # 5.3. Kiểm tra tính dừng bằng ADF
        # Check stationarity for 'VNINDEX' and the generated PCs
        vars_to_check = ["VNINDEX"] + [f"PC{i+1}" for i in range(X_pca_df.shape[1])]  # Check all generated PCs
        adf_results = {}
        needs_differencing = False  # Flag to see if differencing is needed

        for var in vars_to_check:
            data_to_check = df_cleaned["VNINDEX"] if var == "VNINDEX" else X_pca_df[var]
            if not data_to_check.empty:
                try:
                    adf_result = adfuller(data_to_check)
                    print(f"\nKiểm tra ADF cho {var}:")
                    print(f"ADF Statistic: {adf_result[0]:.4f}")
                    print(f"p-value: {adf_result[1]:.4f}")
                    adf_results[var] = adf_result
                    if adf_result[1] < 0.05:
                        print(f"{var} là chuỗi dừng (p-value < 0.05)")
                    else:
                        print(f"{var} không dừng (p-value >= 0.05), cần sai phân.")
                        if var == "VNINDEX":  # Only difference the target if needed
                            needs_differencing = True

                except ValueError as e:
                    print(f"\nError running ADF test for {var}: {e}. Skipping.")

            else:
                print(f"\nData for {var} is empty. Cannot run ADF test.")

        # If VNINDEX is not stationary, difference both VNINDEX and X_pca_df
        if needs_differencing:
            print("\nVNINDEX không dừng, tiến hành sai phân dữ liệu.")
            y_diff = y.diff().dropna()
            X_pca_df_diff = X_pca_df.diff().dropna()
            y_final = y_diff
            X_pca_df_final = X_pca_df_diff.loc[y_final.index]  # Align indices after differencing
        else:
            print("\nVNINDEX đã dừng, không cần sai phân.")
            y_final = y
            X_pca_df_final = X_pca_df.copy()  # Use original PCs if no differencing needed

        ## Bổ sung kiểm định sai phân để xác định tính màu vụ
        # Kiểm tra tính dừng sau sai phân
        print("\nKiểm tra ADF sau sai phân:")
        adf_result_y_diff = adfuller(y_final)
        print(f"ADF Statistic (y_diff): {adf_result_y_diff[0]:.4f}, p-value: {adf_result_y_diff[1]:.4f}")
        for i in range(X_pca_df_final.shape[1]):
            adf_result_pc = adfuller(X_pca_df_final[f"PC{i+1}"])
            print(f"ADF Statistic (PC{i+1}): {adf_result_pc[0]:.4f}, p-value: {adf_result_pc[1]:.4f}")

        ## ======= Bổ sung Chart kiểm định ADF
        # Ensure X_pca_df_final is not empty and has columns befor building ARIMAX
        if not X_pca_df_final.empty and X_pca_df_final.shape[1] > 0:
            # 6. Xây dựng mô hình ARIMAX
            # Check if y_final and X_pca_df_final have the same index length
            if len(y_final) == len(X_pca_df_final):
                train_size = int(len(y_final) * 0.8)
                y_train = y_final[:train_size]
                y_test = y_final[train_size:]
                X_train_pca = X_pca_df_final[:train_size]
                X_test_pca = X_pca_df_final[train_size:]

                print(f"\nShape of y_train: {y_train.shape}")
                print(f"Shape of y_test: {y_test.shape}")
                print(f"Shape of X_train_pca: {X_train_pca.shape}")
                print(f"Shape of X_test_pca: {X_test_pca.shape}")

                ## ==== Cố định SARIAX(p, d, q) = (1, 0/1, 1)
                # Determine the 'd' order for SARIMAX. If differenced, d=0. If not differenced and originally stationary, d=0. If not differenced and not stationary, d=1 (should be handled by differencing step).
                # Assuming the differencing step handles non-stationarity
                sarimax_p_order =0
                sarimax_d_order =1
                sarimax_q_order =1

                # Check if train data is sufficient
                if (len(y_train) > 0 and len(X_train_pca) > 0 and len(y_train) == len(X_train_pca)):
                    # Huấn luyện #1 mô hình ARIMAX (p=1, d=0 , q=1). Use d=0 if differenced manually.
                    # Huấn luyện #2 mô hình ARIMAX (p=1, 1, q=1).
                    # Huấn luyện #3 mô hình ARIMAX(1, 1, 0)
                    try:
                        model = SARIMAX(
                            y_train,
                            exog=X_train_pca,
                            order=(sarimax_p_order, sarimax_d_order, sarimax_q_order), ## change 0 => cho huấn luyện #3
                            enforce_stationarity=False,
                            enforce_invertibility=True,
                        )  # Added enforce=False for robustness
                        arimax_model = model.fit(disp=False)
                        print("\nTóm tắt mô hình ARIMAX:")
                        print(arimax_model.summary())

                        # Dự báo
                        if (len(y_test) > 0 and len(X_test_pca) > 0 and len(y_test) == len(X_test_pca)):
                            y_pred = arimax_model.forecast(steps=len(y_test), exog=X_test_pca)
                            # Ensure y_test and y_pred are aligned before calculating RMSE
                            if len(y_test) == len(y_pred):
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                print(f"\nRMSE trên tập test: {rmse:.4f}")
                            else:
                                print("\nPrediction length does not match test data length. Cannot calculate RMSE.")
                        else:
                            print("\nTest data (y_test or X_test_pca) is empty or mismatched length. Skipping forecasting.")
                    except Exception as e:
                        print(f"\nError fitting or forecasting ARIMAX model: {e}")
                        print("Check your ARIMAX order (p,d,q) and the stationarity")

                #print(f"\nTham số ARIMAX tối ưu (p, d, q): {best_pdq} với AIC: {best_aic}")
                ##--- end tìm tham so toi ưu
                # Kiểm tra mùa vụ trên y_diff
                plt.figure(figsize=(12, 6))
                plot_acf(y_diff, lags=252)
                plt.title("ACF of VNINDEX after Differencing")
                plt.show()
                plt.figure(figsize=(12, 6))
                plot_pacf(y_diff, lags=252)
                plt.title("PACF of VNINDEX after Differencing")
                plt.show()

                # --- added Kiểm tra StartDate/Endate Report sample
                print(f"\nStart date of training data: {y_train.index.min()}")
                print(f"End date of training data: {y_train.index.max()}")
                try:
                    model = SARIMAX(
                        y_train,
                        exog=X_train_pca,
                        order=(sarimax_p_order, sarimax_d_order, sarimax_q_order),
                        enforce_stationarity=True,
                        enforce_invertibility=True
                    )
                    arimax_model = model.fit(disp=False)
                    print("\nTóm tắt mô hình ARIMAX:")
                    print(arimax_model.summary())
                    print(f"\nRMSE trên tập train: {np.sqrt(mean_squared_error(y_train, arimax_model.fittedvalues)):.4f}")

                    # Dự báo
                    if len(y_test) > 0 and len(X_test_pca) > 0 and len(y_test) == len(X_test_pca):
                        y_pred = arimax_model.forecast(steps=len(y_test), exog=X_test_pca)
                        # Ensure y_test and y_pred are aligned before calculating RMSE
                        if len(y_test) == len(y_pred):
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            print(f"\nRMSE trên tập test: {rmse:.4f}")
                        else:
                            print("\nPrediction length does not match test data length. Cannot calculate RMSE.")
                    else:
                        print(
                            "\nTest data (y_test or X_test_pca) is empty or mismatched length. Skipping forecasting."
                        )
                except Exception as e:
                    print(f"\nError fitting or forecasting ARIMAX model: {e}")
                    print("Check your ARIMAX order (p,d,q) and the stationarity")
##=============== Optimize chuyển kiểm tra Overfitting lên trước khi quyết định giảm chiều =====

# 7. Dự báo cho các ngày trong tương lai (2017-05-29, 2017-05-30, 2017-05-31)
data = pd.read_csv("VNINDEX.csv", index_col="Day", parse_dates=True)
y_full = data['VNINDEX']
future_dates = pd.to_datetime(['2017-05-29', '2017-05-30', '2017-05-31'])
X_future_raw = data.loc[data.index.isin(future_dates)]  # Giữ nguyên tất cả cột, bao gồm VNINDEX
y_future_actual = y_full.loc[y_full.index.isin(future_dates)]

# Kiểm tra số hàng của X_future_raw
print(f"\nSố hàng của X_future_raw: {X_future_raw.shape[0]} (nên là 3)")
if X_future_raw.shape[0] != 3:
    print("Cảnh báo: Số hàng của X_future_raw không khớp với 3 ngày dự báo.")
    print("Chỉ số của X_future_raw:", X_future_raw.index.tolist())

# Chuẩn hóa dữ liệu 3 ngày bằng cùng scaler
X_future_scaled = scaler.transform(X_future_raw)
X_future_scaled_df = pd.DataFrame(X_future_scaled, columns=X_future_raw.columns, index=X_future_raw.index)

# Loại bỏ cột VNINDEX sau khi chuẩn hóa
if 'VNINDEX' in X_future_scaled_df.columns:
    X_future_scaled_df = X_future_scaled_df.drop(columns=['VNINDEX'])

# Apply the same cleaning steps to the future data
if 'noisy_variables' in locals() and noisy_variables:
    cols_to_drop_noise_future = [col for col in noisy_variables if col in X_future_scaled_df.columns]
    X_future_cleaned_noise = X_future_scaled_df.drop(columns=cols_to_drop_noise_future)
    print(f"Removed noisy variables from future data: {cols_to_drop_noise_future}")
else:
    X_future_cleaned_noise = X_future_scaled_df.copy()

if 'variables_to_drop_redundant' in locals() and variables_to_drop_redundant:
    cols_to_drop_redundant_future = [col for col in variables_to_drop_redundant if col in X_future_cleaned_noise.columns]
    X_future_cleaned = X_future_cleaned_noise.drop(columns=cols_to_drop_redundant_future)
    print(f"Removed redundant variables from future data: {cols_to_drop_redundant_future}")
else:
    X_future_cleaned = X_future_cleaned_noise.copy()

# Kiểm tra khớp cột với X_cleaned
if set(X_future_cleaned.columns) != set(X_cleaned.columns):
    missing_cols = set(X_cleaned.columns) - set(X_future_cleaned.columns)
    extra_cols = set(X_future_cleaned.columns) - set(X_cleaned.columns)
    print(f"Cảnh báo: Các cột không khớp giữa X_future_cleaned và X_cleaned!")
    print(f"Thiếu cột trong X_future_cleaned: {missing_cols}")
    print(f"Cột thừa trong X_future_cleaned: {extra_cols}")
    raise ValueError("Feature mismatch between X_future_cleaned and X_cleaned.")

# Kiểm tra số hàng của X_future_cleaned
print(f"Số hàng của X_future_cleaned: {X_future_cleaned.shape[0]} (nên là 3)")
if X_future_cleaned.shape[0] != 3:
    print("Cảnh báo: Số hàng của X_future_cleaned không khớp với 3 ngày dự báo.")
    print("Chỉ số của X_future_cleaned:", X_future_cleaned.index.tolist())

# Transform the cleaned future data using pca_final
X_future_pca = pca_final.transform(X_future_cleaned)
X_future_pca_df = pd.DataFrame(
    X_future_pca, columns=[f"PC{i+1}" for i in range(n_components_final_pca)], index=future_dates
)

# Kiểm tra kích thước của X_future_pca_df
print(f"Kích thước của X_future_pca_df: {X_future_pca_df.shape} (nên là 3 hàng, {n_components_final_pca} cột)")
print("Chỉ số của X_future_pca_df:", X_future_pca_df.index.tolist())

# Dự báo
try:
    if 'arimax_model' in locals() and arimax_model is not None:
        plt.figure(figsize=(12, 6))
        future_forecast_diff= arimax_model.forecast(steps=len(future_dates), exog=X_future_pca_df)

        # Khôi phục giá trị gốc nếu đã sai phân
        if needs_differencing:
            last_value = y_full[y_full.index < future_dates[0]].iloc[-1]
            future_forecast_undiff = [last_value] + [last_value + future_forecast_diff[:i].sum() for i in range(1, len(future_forecast_diff)+1)]
            future_forecast = pd.Series(future_forecast_undiff[1:], index=future_dates)
        else:
            future_forecast = future_forecast_diff

        future_results = pd.DataFrame({
            'Date': future_dates,
            'Actual_VNINDEX': y_future_actual,
            'Predicted_VNINDEX': future_forecast,
            'Error': abs(y_future_actual - future_forecast)
        })
        print("\nDự báo, giá trị thực tế và sai số VNINDEX cho các ngày 2017-05-29, 2017-05-30, 2017-05-31:")
        print(future_results.to_string(index=False))
    else:
        print("\nLỗi: Mô hình ARIMAX chưa được huấn luyện. Kiểm tra lại bước huấn luyện mô hình.")
        raise ValueError("ARIMAX model not found or not fitted.")
except Exception as e:
    print(f"\nLỗi khi dự báo: {e}")
    print("Kiểm tra mô hình ARIMAX hoặc dữ liệu exog.")

# Vẽ biểu đồ thu gọn chỉ năm 2017
try:
    if 'arimax_model' in locals() and arimax_model is not None:
        plt.figure(figsize=(12, 6))

        # Lọc dữ liệu VNINDEX thực tế cho năm 2017
        y_2017 = y_full[y_full.index.year == 2017]
        plt.plot(y_2017.index, y_2017, label='VNINDEX Thực tế (2017)', color='blue', alpha=0.7, linewidth=2)

        # Vẽ các điểm dự đoán
        if 'future_forecast' in locals() and 'y_future_actual' in locals():
            # Nếu đã sai phân, sử dụng future_forecast_undiff (giá trị khôi phục)
            if needs_differencing:
                last_value = y_full[y_full.index < future_dates[0]].iloc[-1]
                future_forecast_undiff = [last_value] + [last_value + future_forecast_diff[:i].sum() for i in range(1, len(future_forecast_diff)+1)]
                forecast_values = future_forecast_undiff[1:]  # Bỏ giá trị đầu tiên (last_value)
            else:
                forecast_values = future_forecast

            # Vẽ các điểm dự đoán
            plt.scatter(future_dates, forecast_values, label='Dự báo VNINDEX', color='red', marker='o', s=100, zorder=5)
            # Vẽ giá trị thực tế cho 3 ngày để so sánh
            plt.scatter(future_dates, y_future_actual, label='VNINDEX Thực tế (Dự báo)', color='green', marker='x', s=100, zorder=5)

            # Thêm nhãn giá trị cho các điểm dự đoán
            for i, (date, pred, actual) in enumerate(zip(future_dates, forecast_values, y_future_actual)):
                plt.text(date, pred, f'{pred:.2f}', fontsize=9, ha='right', va='bottom', color='red')
                plt.text(date, actual, f'{actual:.2f}', fontsize=9, ha='left', va='top', color='green')

        plt.xlabel('Ngày', fontsize=12)
        plt.ylabel('VNINDEX', fontsize=12)
        plt.title('Dự báo VNINDEX năm 2017 và Kết quả Dự đoán (2017-05-29 đến 2017-05-31)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        # Lưu biểu đồ
        plt.savefig("VNINDEX_forecast_2017.png", dpi=300)
        plt.show()
        plt.close()

        try:
            files.download("VNINDEX_forecast_2017.png")
            print("\nĐã lưu biểu đồ dự báo 2017 vào 'VNINDEX_forecast_2017.png'")
        except NameError:
            print("\nChạy trong môi trường không phải Colab, bỏ qua tải file.")
            print("Biểu đồ đã được lưu cục bộ: VNINDEX_forecast_2017.png")
    else:
        print("\nLỗi: Mô hình ARIMAX chưa được huấn luyện. Không thể vẽ biểu đồ dự báo.")
except Exception as e:
    print(f"\nLỗi khi tạo biểu đồ dự báo: {e}")
    print("Kiểm tra dữ liệu đầu vào (y_full, future_forecast, y_future_actual) hoặc cấu hình matplotlib.")
