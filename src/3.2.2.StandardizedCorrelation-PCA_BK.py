#3.2.2.StandardizedCorrelation-PCA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tabulate import tabulate
import warnings
from google.colab import files # Import files for download

warnings.filterwarnings('ignore')

# 1.1 Đọc dữ liệu & Chuẩn hoá
# Use a different variable name for the initial read to avoid confusion with the standardized df
df_raw = pd.read_csv('VNINDEX_iqr_filled_HSG_Adjust.csv', parse_dates=[0], index_col=0)

# Chuẩn hóa z-score (bao gồm mean centering)
scaler = StandardScaler()
# Fit and transform the raw data
df_standardized = pd.DataFrame(scaler.fit_transform(df_raw), columns=df_raw.columns, index=df_raw.index)

# Save and download standardized data (assuming Colab environment)
try:
    df_standardized.to_csv('VNINDEX_standardized.csv')
    files.download('VNINDEX_standardized.csv')
    print("Đã lưu dữ liệu chuẩn hóa vào VNINDEX_standardized.csv")
except NameError:
    print("Chạy trong môi trường không phải Colab, bỏ qua download file.")

# 1.2 Kiểm tra trung bình và độ lệch chuẩn
means = df_standardized.mean()
stds = df_standardized.std()
print("Trung bình của dữ liệu đã chuẩn hóa:")
print(means)
print("\nĐộ lệch chuẩn của dữ liệu đã chuẩn hóa:")
print(stds)


# 2 Tính toán ma trận hiệp phương sai và ma trận tương quan Pearson
# Tách biến mục tiêu trước khi tính ma trận hiệp phương sai và tương quan
if 'VNINDEX' in df_standardized.columns:
    X_Variant = df_standardized.drop(columns=['VNINDEX'])
    y_Target = df_standardized['VNINDEX'] # Keep target for later
else:
    print("Warning: 'VNINDEX' column not found in the standardized DataFrame.")
    raise ValueError("'VNINDEX' column not found after standardization. Cannot proceed.")


# 2.2 Tính Ma trận Hiệp phương sai từ dữ liệu đã chuẩn hóa
covariance_matrix = X_Variant.cov()

# 2.3 Tính Ma trận Tương quan
correlation_matrix = X_Variant.corr()

print("Ma trận hiệp phương sai (5x5 đầu tiên):")
print(covariance_matrix.iloc[:5, :5])
print("\nMa trận tương quan Pearson (5x5 đầu tiên):")
print(correlation_matrix.iloc[:5, :5])

# So sánh hai ma trận
are_matrices_close = np.allclose(covariance_matrix, correlation_matrix, atol=1e-5)
print(f"\nMa trận Hiệp phương sai và Ma trận Tương quan có gần giống nhau không (sử dụng np.allclose)? {are_matrices_close}")

# Save and download matrices (assuming Colab environment)
try:
    correlation_matrix.to_csv('VNINDEX_correlation_matrix.csv')
    covariance_matrix.to_csv('VNINDEX_covariance_matrix.csv')
    files.download('VNINDEX_correlation_matrix.csv')
    files.download('VNINDEX_covariance_matrix.csv')
    print("\nĐã lưu ma trận tương quan vào VNINDEX_correlation_matrix.csv")
    print("Đã lưu ma trận hiệp phương sai VNINDEX_covariance_matrix.csv")
except NameError:
    print("\nRunning in a non-Colab environment, skipping file downloads.")
    print("Files saved locally: VNINDEX_correlation_matrix.csv, VNINDEX_covariance_matrix.csv")


print("\nMa trận Tương quan Pearson:\n")
# Trực quan hóa Ma trận Tương quan bằng Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Heatmap Ma trận Tương quan Pearson (Không bao gồm VNINDEX)')
plt.tight_layout()
plt.savefig('Correlation_heatmap.png')
plt.show()
plt.close()
try:
    files.download('Correlation_heatmap.png')
    print("\nĐã lưu heatmap ma trận tương quan vào 'correlation_heatmap.png'")
except NameError:
    print("\nRunning in a non-Colab environment, skipping file downloads.")
    print("Heatmap saved locally: Correlation_heatmap.png")


# 2.5. Đánh giá Mối quan hệ Tương quan (trên X_Variant)
high_corr_threshold = 0.8
high_corr_pairs = {}
totalVariant = 0
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.iloc[i, j]
        totalVariant = totalVariant + 1
        if abs(corr_value) >= high_corr_threshold:
            high_corr_pairs[(col1, col2)] = corr_value

if high_corr_pairs:
    print(f"\nTổng số cặp biến :{len(high_corr_pairs)} có |r| >= {high_corr_threshold} trên tổng số cặp biến: {totalVariant}")
    print(f"\nCác cặp biến có tương quan |r| >= {high_corr_threshold}:")
    for pair, corr in high_corr_pairs.items():
        print(f"  {pair[0]} và {pair[1]}: {corr:.4f}")
    print("\nNhận xét: Các cặp biến này có mối quan hệ tuyến tính mạnh, có thể chứa thông tin trùng lặp.")
    print("PCA sẽ hiệu quả trong việc kết hợp các biến này thành ít thành phần hơn.")
else:
    print(f"\nKhông tìm thấy cặp biến nào có tương quan |r| >= {high_corr_threshold}.")
    print("Nhận xét: Các biến có vẻ tương đối độc lập theo nghĩa tuyến tính. PCA vẫn có thể giúp tìm các thành phần chính, nhưng mức độ giảm chiều có thể không lớn nếu không có nhóm biến tương quan cao.")

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
    print("\nNhận xét: Các cặp biến này có mối quan hệ tuyến tính yếu. Chúng có thể đại diện cho các yếu tố độc lập trong dữ liệu.")
else:
    print(f"\nKhông tìm thấy cặp biến nào có tương quan |r| < {low_corr_threshold}.")

# 2.5.4 Kiểm tra phương sai để xác định ngưỡng (trên X_Variant)
variances = np.diag(covariance_matrix)
percentile_threshold = 5
print(f"\nPhương sai nhỏ nhất (trên X_Variant): {variances.min():.4f}")
print(f"Phương sai trung bình (trên X_Variant): {variances.mean():.4f}")
print(f"Phân vị {percentile_threshold}% của variances (trên X_Variant): {np.percentile(variances, percentile_threshold):.4f}")

# Set low variance threshold - adjust based on percentile if needed
# Example: keep variables with variance > 5th percentile
low_variance_threshold = np.percentile(variances, percentile_threshold)
print(f"Ngưỡng phương sai thấp được sử dụng (dựa trên {percentile_threshold}% phân vị): {low_variance_threshold:.4f}")


# 3. Dùng pca.fit(X_Variant) để phân tích ban đầu - Chỉ huấn luyện mô hình
pca = PCA()
if not X_Variant.empty:
    pca.fit(X_Variant)

    #3.1 Tính eigenvalues
    eigenvalues = pca.explained_variance_
    print("\nEigenvalues (phương sai của từng PC):", eigenvalues[:min(5, len(eigenvalues))])
    print("Ý nghĩa: Eigenvalues thể hiện phương sai mà mỗi thành phần chính giải thích.")

    # 3.2 Tính eigenvectors
    eigenvectors = pca.components_
    print("\nEigenvectors (hướng của PC1):", eigenvectors[0][:min(5, len(eigenvectors[0]))])
    print("Ý nghĩa: Eigenvectors định hướng cho mỗi PC, thể hiện mức độ đóng góp của từng biến gốc.")

    # 3.3 Tính tỷ lệ phương sai giải thích
    explained_variance_ratio = pca.explained_variance_ratio_
    print("\nTỷ lệ phương sai giải thích:", explained_variance_ratio[:min(5, len(explained_variance_ratio))])

    # 3.4 Tính tỷ lệ phương sai tích lũy
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print("\nTỷ lệ phương sai tích lũy:", cumulative_variance[:min(5, len(cumulative_variance))])
    print("Ý nghĩa: Tỷ lệ phương sai tích lũy cho biết tổng phương sai được giải thích.")

    # 3.5.1 Đánh giá dựa trên phương sai giải thích
    variance_threshold = 0.90
    n_components_to_keep = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"\n--- Đánh giá PCA ---")
    print(f"Tổng số biến gốc (sau tách VNINDEX): {X_Variant.shape[1]}")
    print(f"Ngưỡng phương sai tích lũy mong muốn: {variance_threshold*100:.0f}%")
    print(f"Số thành phần chính PCA có thể giữ lại: {n_components_to_keep}")
    print(f"\n--- Đánh giá PCA tiêu chí 1 ---")

    if n_components_to_keep <= X_Variant.shape[1]:
         print(f"Để giải thích ít nhất {variance_threshold*100:.0f}% phương sai, cần {n_components_to_keep} thành phần chính.")
         reduction_percentage = (1 - n_components_to_keep / X_Variant.shape[1]) * 100
         print(f"Có thể giảm chiều dữ liệu từ {X_Variant.shape[1]} xuống {n_components_to_keep} chiều (giảm khoảng {reduction_percentage:.2f}%).")
         print("\nNhận xét: PCA cho phép giảm đáng kể số chiều dữ liệu trong khi vẫn giữ lại phần lớn thông tin.")
    else:
         print(f"Ngưỡng phương sai {variance_threshold*100:.0f}% không thể đạt được ngay cả với tất cả các thành phần.")
         print(f"Tổng phương sai giải thích tối đa là {cumulative_variance[-1]*100:.2f}%.")
         print("Nhận xét: Việc giảm chiều bằng PCA không hiệu quả nhiều trong trường hợp này nếu mục tiêu là giữ lại tỷ lệ phương sai rất cao.")

    print(f"\n--- Đánh giá PCA tiêu chí 2 ---")
    if cumulative_variance[-1] >= variance_threshold:
        print(f"Để giải thích ít nhất {variance_threshold*100:.0f}% phương sai, cần {n_components_to_keep} thành phần chính.")
        reduction_percentage = (1 - n_components_to_keep / X_Variant.shape[1]) * 100
        print(f"Có thể giảm chiều dữ liệu từ {X_Variant.shape[1]} xuống {n_components_to_keep} chiều (giảm khoảng {reduction_percentage:.2f}%).")
        print("\nNhận xét: PCA cho phép giảm đáng kể số chiều dữ liệu trong khi vẫn giữ lại phần lớn thông tin.")
    else:
         print(f"Ngay cả khi giữ lại tất cả {X_Variant.shape[1]} thành phần, tổng phương sai giải thích là {cumulative_variance[-1]*100:.2f}%.")
         print("Nhận xét: Có thể tập dữ liệu này không chứa nhiều thông tin dư thừa (biến tương quan cao).")

    # 3.5.2 Đánh giá: 5 thành phần chính có đủ tốt không?
    print(f"\n--- Đánh khả năng giảm chiều các thành phần chính PCA. : ")
    if len(cumulative_variance) > 4: # Check if 5th PC (index 4) exists
        if cumulative_variance[4] >= variance_threshold:
            print(f"5 thành phần chính giải thích {cumulative_variance[4]*100:.2f}% phương sai, đủ tốt để giảm chiều cho ARIMAX.")
        else:
            print(f"5 thành phần chính chỉ giải thích {cumulative_variance[4]*100:.2f}% phương sai, có thể cần thêm PC.")
    else:
         print(f"Chỉ có {len(cumulative_variance)} thành phần chính. Tổng phương sai giải thích là {cumulative_variance[-1]*100:.2f}%.")

    print(f"\n--- Kết luận khả năng giảm chiều PCA ---")
    print(f"Tổng số biến gốc: {df_standardized.shape[1]}") # Includes VNINDEX here
    print(f"Ngưỡng phương sai tích lũy mong muốn: {variance_threshold*100:.0f}%")
    if len(cumulative_variance) > 4:
        print(f"Ngưỡng phương sai tích lũy đạt được (với 5 PC): {cumulative_variance[4]*100:.0f}%")
    else:
        print("Không đủ 5 thành phần chính.")

    print(f"Số thành phần chính PCA có thể giữ lại theo ngưỡng {variance_threshold*100:.0f}%: {n_components_to_keep}")

    #3.6 Biểu đồ tỷ lệ phương sai tích lũy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Số lượng thành phần chính')
    plt.ylabel('Tổng Phương sai Giải thích (%)')
    plt.title('Biểu đồ Phương sai Giải thích Tích lũy của PCA (trên X_Variant)')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.show()
    plt.close()
    try:
        files.download('pca_explained_variance.png')
        print("\nĐã lưu biểu đồ phương sai giải thích tích lũy vào 'pca_explained_variance.png'")
    except NameError:
        print("\nRunning in a non-Colab environment, skipping file downloads.")
        print("Plot saved locally: pca_explained_variance.png")


    # 4. Tính toán và lọc dữ liệu dựa trên ma trận hiệp phương sai, ma trận tương quan và loadings
    # Tính vector tương quan (loadings) trên X_Variant
    if pca.n_components_ > 0:
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(loadings.shape[1])], index=X_Variant.columns)
        print("\nLoadings cho PC1 (đã được sắp xếp giảm dần):")
        if 'PC1' in loadings_df.columns:
             print(loadings_df['PC1'].sort_values(ascending=False).head(min(5, len(loadings_df['PC1']))))
             print("Ý nghĩa: Loadings thể hiện mức độ đóng góp của các biến gốc vào PC1.")
        else:
             print("PC1 does not exist in loadings_df. Cannot print loadings.")
    else:
         print("\nPCA resulted in 0 components on X_Variant. Cannot calculate loadings.")


    # 4.1 Loại nhiễu: Dựa trên ma trận hiệp phương sai (phương sai thấp) và loadings (đóng góp thấp)
    # Identify low variance variables based on the calculated threshold
    low_variance_vars = covariance_matrix.columns[variances < low_variance_threshold].tolist()
    print(f"\nBiến có phương sai thấp (dựa trên ngưỡng {low_variance_threshold:.4f}): {low_variance_vars}")

    # Identify low contribution variables based on loadings (on X_Variant)
    # Determine a threshold based on the distribution of mean loadings
    if 'loadings_df' in locals() and not loadings_df.empty:
        mean_loadings = np.abs(loadings_df).mean(axis=1)
        # Example: Keep top 80% variables by mean loading
        percentile_loading_threshold = 20 # Keep variables above the 20th percentile of mean loadings
        low_contribution_threshold = np.percentile(mean_loadings, percentile_loading_threshold)
        print(f"Ngưỡng đóng góp thấp được sử dụng (dựa trên {percentile_loading_threshold}th percentile của mean loadings): {low_contribution_threshold:.4f}")

        low_contribution_vars = mean_loadings[mean_loadings < low_contribution_threshold].index.tolist()
        print(f"Số biến có mean_loadings < {low_contribution_threshold:.4f}: {len(low_contribution_vars)}")
        print(f"Danh sách mean_loadings thấp nhất:\n{mean_loadings.sort_values().head(10)}") # Print more if needed
    else:
         print("loadings_df is not available or is empty. Skipping low contribution variable check.")
         low_contribution_vars = [] # Ensure this list is empty if loadings_df is not available

    # Combine low variance and low contribution variables to identify noise
    noisy_variables = list(set(low_variance_vars + low_contribution_vars))
    print(f"\nBiến nhiễu bị loại bỏ: {noisy_variables}")

    # Create df_cleaned_noise by dropping noisy variables from df_standardized
    df_cleaned_noise = df_standardized.drop(columns=noisy_variables)
    print(f"Số lượng biến sau khi lọc nhiễu: {df_cleaned_noise.shape[1]}") # Includes VNINDEX potentially

    # 4.2 Loại bỏ biến dư thừa: Dựa trên ma trận tương quan Pearson (tương quan > 0.8) on df_cleaned_noise
    variables_to_drop_redundant = set()
    # Only proceed if there are enough columns for correlation matrix after noise cleaning
    if not df_cleaned_noise.empty and df_cleaned_noise.shape[1] > 1:
        # Ensure 'VNINDEX' is excluded from redundant variable check
        df_cleaned_noise_features = df_cleaned_noise.drop(columns=['VNINDEX']) if 'VNINDEX' in df_cleaned_noise.columns else df_cleaned_noise.copy()

        if not df_cleaned_noise_features.empty and df_cleaned_noise_features.shape[1] > 1:
            corr_matrix_cleaned = df_cleaned_noise_features.corr()
            highly_correlated_pairs_cleaned = []
            for i in range(len(corr_matrix_cleaned.columns)):
                for j in range(i + 1, len(corr_matrix_cleaned.columns)):
                    if abs(corr_matrix_cleaned.iloc[i, j]) > high_corr_threshold: # Use the high_corr_threshold defined earlier
                        highly_correlated_pairs_cleaned.append((corr_matrix_cleaned.columns[i], corr_matrix_cleaned.columns[j], corr_matrix_cleaned.iloc[i, j]))

            for var1, var2, corr in highly_correlated_pairs_cleaned:
                 # Decide which one to drop - here we drop var2 from the redundant pair
                 # Ensure the variable is not the target variable
                 if var1 != 'VNINDEX' and var2 != 'VNINDEX':
                      # To avoid adding the same variable multiple times if it's correlated with several others
                      if var2 not in variables_to_drop_redundant:
                          variables_to_drop_redundant.add(var2)

            df_cleaned = df_cleaned_noise.drop(columns=list(variables_to_drop_redundant))
            print(f"\nBiến dư thừa bị loại bỏ (dựa trên ma trận Pearson trên dữ liệu đã lọc nhiễu): {list(variables_to_drop_redundant)}")
        else:
             print("\nKhông đủ biến (sau lọc nhiễu và loại VNINDEX) để kiểm tra biến dư thừa.")
             df_cleaned = df_cleaned_noise.copy() # Keep the same DataFrame
    else:
         print("\ndf_cleaned_noise is empty or has only one column. Skipping redundant variable check.")
         df_cleaned = df_cleaned_noise.copy() # Keep the same DataFrame

else: # Case where initial X_Variant was empty
    print("Initial X_Variant DataFrame is empty. Cannot proceed with PCA analysis and filtering.")
    # Set df_cleaned to an empty state or handle appropriately
    df_cleaned = pd.DataFrame(index=df_standardized.index)
    explained_variance_ratio = np.array([])
    cumulative_variance = np.array([])
    n_components_to_keep = 0
    # Skip subsequent steps that depend on PCA and filtering

# 5. Pha giảm chiều  Áp dụng PCA trên dữ liệu đã lọc
# Ensure df_cleaned has columns and contains VNINDEX before proceeding
X_cleaned = pd.DataFrame()
y = pd.Series()
X_pca_df = pd.DataFrame() # Initialize X_pca_df

if 'VNINDEX' in df_cleaned.columns and df_cleaned.shape[1] > 1: # Need VNINDEX and at least one feature for PCA
    X_cleaned = df_cleaned.drop(columns=['VNINDEX'])
    y = df_cleaned['VNINDEX']
    print(f"\nShape of X_cleaned before final PCA: {X_cleaned.shape}")

    # Check if X_cleaned is empty before applying final PCA
    if not X_cleaned.empty and X_cleaned.shape[1] > 0:
         # Ensure n_components_to_keep is valid for X_cleaned dimensions
        # Decide on the number of components for the final PCA
        # Option 1: Use n_components_to_keep from initial PCA based on cumulative variance threshold
        # Option 2: Choose a fixed number, e.g., 5, if that's the project requirement
        # Let's use 5 as the user's last code snippet intended, but ensure it doesn't exceed the number of features
        n_components_final_pca = min(5, X_cleaned.shape[1]) # Use 5 PCs, but no more than available features

        if n_components_final_pca > 0:
             print(f"Applying final PCA with n_components = {n_components_final_pca}")
             pca_final = PCA(n_components=n_components_final_pca)
             X_pca_final = pca_final.fit_transform(X_cleaned)

             # Tạo DataFrame cho các thành phần chính
             X_pca_df = pd.DataFrame(X_pca_final, columns=[f'PC{i+1}' for i in range(n_components_final_pca)], index=df_cleaned.index)

             # Tính lại tỷ lệ phương sai tích lũy sau khi lọc
             explained_variance_ratio_final = pca_final.explained_variance_ratio_ * 100
             cumulative_variance_final = np.cumsum(explained_variance_ratio_final)

             print("\nTỷ lệ phương sai giải thích và tích lũy sau khi lọc:")
             for i in range(n_components_final_pca):
                  print(f"PC{i+1}: {explained_variance_ratio_final[i]:.2f}% ({cumulative_variance_final[i]:.2f}% tích lũy)")

             if cumulative_variance_final[-1] >= 80: # Check the last cumulative variance against a reasonable threshold
               print(f"Ý nghĩa Tỷ lệ phương sai tích lũy {cumulative_variance_final[-1]:.2f}% trên dữ liệu đã lọc và giảm chiều là phù hợp để sử dụng trong ARIMAX.")
             else:
               print(f"Ý nghĩa Tỷ lệ phương sai tích lũy {cumulative_variance_final[-1]:.2f}% sau lọc/giảm chiều thấp hơn 80%.")


             # 6. Kiểm tra tính dừng bằng ADF và kiểm tra kỹ
             # Check stationarity for 'VNINDEX' and the generated PCs
             vars_to_check = ['VNINDEX'] + [f'PC{i+1}' for i in range(X_pca_df.shape[1])] # Check all generated PCs
             adf_results = {}
             needs_differencing = False # Flag to see if differencing is needed

             for var in vars_to_check:
                 data_to_check = df_cleaned['VNINDEX'] if var == 'VNINDEX' else X_pca_df[var]
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
                                  if var == 'VNINDEX': # Only difference the target if needed
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
                 X_pca_df_final = X_pca_df_diff.loc[y_final.index] # Align indices after differencing
             else:
                 print("\nVNINDEX đã dừng, không cần sai phân.")
                 y_final = y
                 X_pca_df_final = X_pca_df.copy() # Use original PCs if no differencing needed

             # Ensure X_pca_df_final is not empty and has columns before building ARIMAX
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


                     # Determine the 'd' order for SARIMAX. If differenced, d=0. If not differenced and originally stationary, d=0. If not differenced and not stationary, d=1 (should be handled by differencing step).
                     sarimax_d_order = 0 # Assuming the differencing step handles non-stationarity

                     # Check if train data is sufficient
                     if len(y_train) > 0 and len(X_train_pca) > 0 and len(y_train) == len(X_train_pca):
                         # Huấn luyện mô hình ARIMAX (p=1, d=0 or 1, q=1). Use d=0 if differenced manually.
                         try:
                             model = SARIMAX(y_train, exog=X_train_pca, order=(1, sarimax_d_order, 1), enforce_stationarity=False, enforce_invertibility=False) # Added enforce=False for robustness
                             arimax_model = model.fit(disp=False)
                             print("\nTóm tắt mô hình ARIMAX:")
                             print(arimax_model.summary())

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
                                  print("\nTest data (y_test or X_test_pca) is empty or mismatched length. Skipping forecasting.")
                         except Exception as e:
                              print(f"\nError fitting or forecasting ARIMAX model: {e}")
                              print("Check your ARIMAX order (p,d,q) and the stationarity")
