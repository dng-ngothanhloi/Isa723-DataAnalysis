##3.2.1.3CompareIQRZ-Score.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_csv("VNINDEX_iqr_filled_HSG_Adjust.csv", index_col="Day", parse_dates=True)
y = data['VNINDEX']
X = data.drop(columns=['VNINDEX'])
## ======= Xoá đã hoàn thành trước đó ======
# Hàm xử lý ngoại lai với IQR (hệ số 3.0)
#def remove_outliers_iqr(df, multiplier=3.0):
#    df_cleaned = df.copy()
#    # Create a boolean mask to identify rows that are NOT outliers in ANY column
#    not_outlier_mask = pd.Series(True, index=df_cleaned.index)
#    for column in df_cleaned.columns:
#        Q1 = df_cleaned[column].quantile(0.25)
#        Q3 = df_cleaned[column].quantile(0.75)
#        IQR = Q3 - Q1
#        lower_bound = Q1 - multiplier * IQR
#        upper_bound = Q3 + multiplier * IQR
#        # Update the mask: keep only rows where the current column value is within bounds
#        not_outlier_mask &= (df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)
#    # Apply the mask to filter the DataFrame
#    df_cleaned = df_cleaned[not_outlier_mask]
#    return df_cleaned

# Hàm tính ma trận hiệp phương sai và tương quan
def compute_matrices(df):
    covariance_matrix = df.cov()
    correlation_matrix = df.corr()
    return covariance_matrix, correlation_matrix

# Hàm kiểm tra ADF
def check_adf(series):
    result = adfuller(series)
    return result[0], result[1]

# Hàm huấn luyện và đánh giá SARIMAX
def evaluate_sarimax(X_scaled, y, train_size=0.8, order=(2, 1, 2)):
    n = int(len(y) * train_size)
    y_train, y_test = y[:n], y[n:]
    X_train, X_test = X_scaled[:n], X_scaled[n:]
    
    # Giảm chiều với PCA
    pca = PCA(n_components=7)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Huấn luyện SARIMAX
    model = SARIMAX(y_train, exog=X_train_pca, order=order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # Dự báo
    y_pred = results.forecast(steps=len(y_test), exog=X_test_pca)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    return rmse, pca.explained_variance_ratio_.sum(), results.aic

# So sánh hai phương pháp
methods = {
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler()
}

results = {}
for method_name, scaler in methods.items():
    # Xử lý ngoại lai cho RobustScaler
    #X_cleaned = remove_outliers_iqr(X) 
    #if method_name == "RobustScaler" else X.copy()
    if method_name == "RobustScaler":
        ##remove_outliers_iqr(X)
        X_processed = X
        # Also filter y to match the rows removed from X
        y_processed = y[y.index.isin(X_processed.index)]
    else: # StandardScaler
        X_processed = X.copy()
        y_processed = y.copy()

    # Chuẩn hóa
    X_scaled = scaler.fit_transform(X_cleaned)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_cleaned.columns, index=X_cleaned.index)
    
    # Tính ma trận
    cov_matrix, corr_matrix = compute_matrices(X_scaled_df)
    
    # Giảm chiều với PCA
    pca = PCA(n_components=7)
    X_pca = pca.fit_transform(X_scaled_df)
    explained_variance_ratio = pca.explained_variance_ratio_.sum()
    
    # Kiểm tra ADF cho VNINDEX và PC1
    adf_y, p_value_y = check_adf(y)
    adf_pc1, p_value_pc1 = check_adf(pd.Series(X_pca[:, 0]))
    
    # Đánh giá SARIMAX
    rmse, total_variance, aic = evaluate_sarimax(X_scaled_df, y)
    
    # Lưu kết quả
    results[method_name] = {
        "Covariance Matrix": cov_matrix,
        "Correlation Matrix": corr_matrix,
        "Explained Variance Ratio": explained_variance_ratio,
        "ADF VNINDEX": (adf_y, p_value_y),
        "ADF PC1": (adf_pc1, p_value_pc1),
        "RMSE": rmse,
        "AIC": aic
    }

# In kết quả
for method_name, result in results.items():
    print(f"\nKết quả với {method_name}:")
    print(f"Tỷ lệ phương sai tích lũy (7 PC): {result['Explained Variance Ratio']:.4f}")
    print(f"ADF VNINDEX: Statistic={result['ADF VNINDEX'][0]:.4f}, p-value={result['ADF VNINDEX'][1]:.4f}")
    print(f"ADF PC1: Statistic={result['ADF PC1'][0]:.4f}, p-value={result['ADF PC1'][1]:.4f}")
    print(f"RMSE trên tập test: {result['RMSE']:.4f}")
    print(f"AIC: {result['AIC']:.4f}")

# Vẽ biểu đồ so sánh RMSE
plt.figure(figsize=(8, 6))
plt.bar(results.keys(), [result['RMSE'] for result in results.values()], color=['blue', 'orange'])
plt.title('So sánh RMSE giữa StandardScaler và RobustScaler')
plt.ylabel('RMSE')
plt.show()