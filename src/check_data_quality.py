import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_excel('6.VNINDEX.xlsx', index_col=0, parse_dates=True)

# 1. Kiểm tra dữ liệu bị thiếu
print("Số giá trị NaN trong mỗi cột:")
print(df.isna().sum())
print("\nSố giá trị 0 trong mỗi cột:")
print((df == 0).sum())

# Kiểm tra ngày bị thiếu
dates = pd.date_range(start='2012-01-03', end='2017-05-31', freq='B')
missing_dates = dates[~dates.isin(df.index)]
print(f"\nSố ngày bị thiếu: {len(missing_dates)}")
if len(missing_dates) > 0:
    print("Các ngày bị thiếu:", missing_dates)

# 2. Kiểm tra dữ liệu ngoại lai (phương pháp IQR)
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers

for column in df.columns:
    outliers = detect_outliers_iqr(df[column])
    if not outliers.empty:
        print(f"\nDữ liệu ngoại lai trong cột {column}:")
        print(outliers)

# Vẽ boxplot để kiểm tra ngoại lai
plt.figure(figsize=(12, 6))
df.boxplot(column=['VNINDEX', 'VN30'])
plt.title('Boxplot VNINDEX và VN30')
plt.savefig('boxplot_vnindex_vn30.png')

# 3. Kiểm tra dữ liệu không nhất quán
# Tính tương quan giữa VNINDEX và VN30
correlation = df['VNINDEX'].corr(df['VN30'])
print(f"\nTương quan giữa VNINDEX và VN30: {correlation}")

# Tính phần trăm thay đổi hàng ngày
df_pct_change = df.pct_change()
outlier_changes = df_pct_change[abs(df_pct_change) > 0.1]  # Biến động >10%
if not outlier_changes.empty:
    print("\nCác ngày có biến động lớn (>10%):")
    print(outlier_changes.dropna(how='all'))

# Vẽ chuỗi thời gian VNINDEX
plt.figure(figsize=(12, 6))
df['VNINDEX'].plot()
plt.title('Chuỗi thời gian VNINDEX')
plt.savefig('timeseries_vnindex.png')