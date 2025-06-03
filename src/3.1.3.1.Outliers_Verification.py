#### 3.1.1.3.Outliers_Verification.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, shapiro
import numpy as np

# Đọc dữ liệu từ file CSV
df = pd.read_csv('VNINDEX.csv', parse_dates=[0], index_col=0)

# Chọn các cột cần phân tích
columns = ['FLC', 'HSG', 'KDC', 'PPC']

# Kiểm tra xem các cột có tồn tại trong dữ liệu
missing_cols = [col for col in columns if col not in df.columns]
if missing_cols:
    print(f"Các cột không tồn tại trong dữ liệu: {missing_cols}")
    exit()

# Bước 1: Vẽ histogram
plt.figure(figsize=(12, 8))
for i, col in enumerate(columns, 1):
    plt.subplot(2, 2, i)
    df[col].dropna().hist(bins=50, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram_FLC_HSG_KDC_PPC.png')
plt.show()
plt.close()
print("Đã lưu histogram vào 'histogram_FLC_HSG_KDC_PPC.png'")

# Bước 2: Vẽ boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[columns].dropna())
plt.title('Boxplot of FLC, HSG, KDC, PPC')
plt.ylabel('Value')
plt.savefig('boxplot_FLC_HSG_KDC_PPC.png')
plt.show()
plt.close()
print("Đã lưu boxplot vào 'boxplot_FLC_HSG_KDC_PPC.png'")

# Bước 3: Kiểm tra phân phối
print("\nPhân tích thống kê cho các cột:")
for col in columns:
    data = df[col].dropna()
    mean = data.mean()
    median = data.median()
    std = data.std()
    skewness = skew(data)
    kurt = kurtosis(data)
    
    # Kiểm định Shapiro-Wilk (kiểm tra tính chuẩn)
    stat, p_value = shapiro(data)
    
    print(f"\nCột {col}:")
    print(f"  Số điểm dữ liệu: {len(data)}")
    print(f"  Trung bình: {mean:.2f}")
    print(f"  Trung vị: {median:.2f}")
    print(f"  Độ lệch chuẩn: {std:.2f}")
    print(f"  Độ lệch (Skewness): {skewness:.2f}")
    print(f"  Độ nhọn (Kurtosis): {kurt:.2f}")
    print(f"  Shapiro-Wilk Test: Statistic={stat:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print("    => Phân phối không chuẩn (p-value < 0.05)")
    else:
        print("    => Không đủ bằng chứng để bác bỏ phân phối chuẩn (p-value >= 0.05)")