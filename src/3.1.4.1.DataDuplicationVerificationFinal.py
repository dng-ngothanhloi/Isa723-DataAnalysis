#####3.2.1.0.DataDuplicationVerificationFinal.py
import pandas as pd
import numpy as np
from datetime import timedelta
from google.colab import files

# 1. Đọc dữ liệu
df_processed = pd.read_csv('VNINDEX_iqr_filled_HSG_Adjust.csv', parse_dates=[0], index_col=0)
df_raw = pd.read_csv('VNINDEX.csv', parse_dates=[0], index_col=0)

# --- Added code to round numeric columns ---
# Chọn tất cả các cột có kiểu dữ liệu số
numeric_cols = df_processed.select_dtypes(include=np.number).columns
# Áp dụng làm tròn 2 chữ số thập phân cho các cột số
df_processed[numeric_cols] = df_processed[numeric_cols].round(2)

numeric_cols = df_raw.select_dtypes(include=np.number).columns
df_raw[numeric_cols] = df_raw[numeric_cols].round(2)
print("Đã làm tròn các cột số đến 2 chữ số thập phân.")
# -------------------------------------------

# 2. So sánh dữ liệu gốc và đã xử lý
print("Processed data shape:", df_processed.shape)
print("Raw data shape:", df_raw.shape)
print("\nMissing values in raw data:\n", df_raw.isna().sum())

# Kiểm tra consecutive duplicates trong raw data
consecutive_duplicates_raw = {}
for col in df_raw.columns:
    if col in df_processed.columns:
        max_consecutive = (df_raw[col].diff() == 0).cumsum().value_counts().max() if not df_raw[col].isna().all() else 0
        consecutive_duplicates_raw[col] = max_consecutive
print("\nMax consecutive duplicates in raw data:")
for col, count in consecutive_duplicates_raw.items():
    print(f"{col}: {count} days")

# 3. Xác định khoảng cách ngày >3 trong raw data
dates_raw = df_raw.index.sort_values()
gaps = []
for i in range(1, len(dates_raw)):
    gap = (dates_raw[i] - dates_raw[i-1]).days
    if gap > 3:  # Khoảng cách >3 ngày
        # Thêm tất cả các ngày trong khoảng cách
        for d in pd.date_range(dates_raw[i-1] + timedelta(days=1), dates_raw[i] - timedelta(days=1)):
            gaps.append(d)

# Loại bỏ các ngày có khoảng cách >3
long_holiday_dates = pd.to_datetime(gaps)
df_cleaned = df_processed[~df_processed.index.isin(long_holiday_dates)]
print(f"\nRemoved {len(df_processed) - len(df_cleaned)} long holiday days")
print("Cleaned data shape:", df_cleaned.shape)

# 4. Xử lý dữ liệu thiếu (chỉ ngày giao dịch)
df_cleaned = df_cleaned.interpolate(method='spline', order=3)
print("\nRemaining missing values:\n", df_cleaned.isna().sum())
df_cleaned = df_cleaned.fillna(df_cleaned.mean())  # Điền NaN bằng trung bình cột

# 5. Kiểm tra consecutive duplicates
consecutive_duplicates = {}
for col in df_cleaned.columns:
    max_consecutive = (df_cleaned[col].diff() == 0).cumsum().value_counts().max()
    consecutive_duplicates[col] = max_consecutive
print("\nMax consecutive duplicates after processing:")
for col, count in consecutive_duplicates.items():
    print(f"{col}: {count} days")

# 6. Loại bỏ cột có lặp quá dài (>20 days)
# to_drop = [col for col, count in consecutive_duplicates.items() if count > 20 and col != 'VNINDEX']
# df_cleaned = df_cleaned.drop(columns=to_drop)
# print(f"\nDropped columns: {to_drop}")

# 7. Lưu dữ liệu
df_cleaned.to_csv('VNINDEX_cleaned_final.csv')
print("\nData saved to 'VNINDEX_cleaned_final.csv'")
files.download('VNINDEX_cleaned_final.csv')