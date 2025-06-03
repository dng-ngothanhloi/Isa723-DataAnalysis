import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Đọc dữ liệu (thay đường dẫn bằng đường dẫn thực tế)
df = pd.read_excel('/path/to/6.VNINDEX.xlsx', index_col=0, parse_dates=True)

# Chọn cột để kiểm tra ngoại lai (ví dụ: VNINDEX)
data = df[['VNINDEX']].copy()

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Áp dụng Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(data_scaled)

# Đánh dấu ngoại lai (-1 là ngoại lai, 1 là bình thường)
df['is_outlier'] = (outlier_labels == -1).astype(int)

# In các điểm ngoại lai
print("Các điểm ngoại lai trong VNINDEX:")
print(df[df['is_outlier'] == 1][['VNINDEX']])

# Xử lý ngoại lai: Thay thế bằng nội suy tuyến tính
df['VNINDEX_cleaned'] = df['VNINDEX'].copy()
df.loc[df['is_outlier'] == 1, 'VNINDEX_cleaned'] = np.nan
df['VNINDEX_cleaned'] = df['VNINDEX_cleaned'].interpolate(method='linear')

# Vẽ biểu đồ so sánh
plt.figure(figsize=(12, 6))
df['VNINDEX'].plot(label='VNINDEX gốc', alpha=0.5)
df['VNINDEX_cleaned'].plot(label='VNINDEX sau xử lý', color='red')
plt.scatter(df[df['is_outlier'] == 1].index, df[df['is_outlier'] == 1]['VNINDEX'], 
            color='black', label='Ngoại lai', marker='o')
plt.title('Chuỗi thời gian VNINDEX với các điểm ngoại lai')
plt.legend()
plt.savefig('isolation_forest_vnindex.png')
plt.show()

# Lưu kết quả để sử dụng trong EViews
df[['VNINDEX', 'is_outlier', 'VNINDEX_cleaned']].to_csv('VNINDEX_outliers.csv')