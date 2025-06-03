import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_excel('6.VNINDEX.xlsx', index_col=0, parse_dates=True)

# 1. Xử lý dữ liệu ngoại lai (IQR)
def detect_and_handle_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    # Thay thế ngoại lai bằng trung vị 5 ngày lân cận
    series_cleaned = series.copy()
    for idx in series[outliers].index:
        window = series.loc[max(series.index[0], idx - pd.Timedelta(days=5)):min(series.index[-1], idx + pd.Timedelta(days=5))]
        series_cleaned.loc[idx] = window.median()
    return series_cleaned, outliers

# Áp dụng cho VNINDEX
df['VNINDEX_cleaned'], outliers = detect_and_handle_outliers(df['VNINDEX'])
print("Dữ liệu ngoại lai trong VNINDEX:")
print(df[outliers]['VNINDEX'])

# Vẽ boxplot
plt.figure(figsize=(12, 6))
df[['VNINDEX', 'VNINDEX_cleaned']].boxplot()
plt.title('Boxplot VNINDEX trước và sau xử lý ngoại lai')
plt.savefig('boxplot_vnindex_cleaned.png')

# 2. Xử lý dữ liệu bị thiếu (Nội suy tuyến tính)
# Kiểm tra NaN
print("\nSố giá trị NaN trong mỗi cột:")
print(df.isna().sum())
# Nội suy tuyến tính
df_interpolated = df.interpolate(method='linear')
print("\nDữ liệu sau nội suy tuyến tính:")
print(df_interpolated.head())

# Kiểm tra ngày bị thiếu
dates = pd.date_range(start='2012-01-03', end='2017-05-31', freq='B')
missing_dates = dates[~dates.isin(df.index)]
if len(missing_dates) > 0:
    print(f"\nSố ngày bị thiếu: {len(missing_dates)}")
    print("Các ngày bị thiếu:", missing_dates)

# 3. Xử lý dữ liệu không nhất quán (Tương quan và ngưỡng biến động)
# Tính tương quan
correlation = df['VNINDEX'].corr(df['VN30'])
print(f"\nTương quan giữa VNINDEX và VN30: {correlation}")

# Kiểm tra biến động lớn
df_pct_change = df.pct_change()
outlier_changes = df_pct_change[abs(df_pct_change) > 0.1]
if not outlier_changes.empty:
    print("\nCác ngày có biến động lớn (>10%):")
    print(outlier_changes.dropna(how='all'))
    # Thay thế bằng nội suy tuyến tính
    for column in outlier_changes.columns:
        df[column] = df[column].interpolate(method='linear')

# Vẽ chuỗi thời gian VNINDEX
plt.figure(figsize=(12, 6))
df['VNINDEX'].plot(label='VNINDEX')
df['VNINDEX_cleaned'].plot(label='VNINDEX Cleaned')
plt.title('Chuỗi thời gian VNINDEX trước và sau xử lý')
plt.legend()
plt.savefig('timeseries_vnindex_cleaned.png')