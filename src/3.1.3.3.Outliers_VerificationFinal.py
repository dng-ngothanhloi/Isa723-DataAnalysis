##### 3.1.4.1.DataDuplicationVerificationFinal.py
import pandas as pd
import numpy as np
from datetime import timedelta
from google.colab import files

# 1. Đọc dữ liệu
df_processed = pd.read_csv('VNINDEX_iqr_filled_HSG_Adjust.csv', parse_dates=[0], index_col=0)
df_raw = pd.read_csv('VNINDEX.csv', parse_dates=[0], index_col=0)

# --- Added code to round numeric columns ---
# Chọn tất cả các cột có kiểu dữ liệu số
#numeric_cols = df_processed.select_dtypes(include=np.number).columns
# Áp dụng làm tròn 2 chữ số thập phân cho các cột số
#df_processed[numeric_cols] = df_processed[numeric_cols].round(6)

#numeric_cols = df_raw.select_dtypes(include=np.number).columns
#df_raw[numeric_cols] = df_raw[numeric_cols].round(6)
#print("Đã làm tròn các cột số đến 2 chữ số thập phân.")
# -------------------------------------------

# 2. So sánh dữ liệu gốc và đã xử lý
print("Raw data shape (VNINDEX.csv):", df_raw.shape)
print("\nMissing values in raw data (VNINDEX.csv):\n", df_raw.isna().sum())

# Kiểm tra consecutive duplicates trong raw data (VNINDEX.csv)
consecutive_duplicates_raw = {}
for col in df_raw.columns:
    if col in df_processed.columns:
        max_consecutive = (df_raw[col].diff() == 0).cumsum().value_counts().max() if not df_raw[col].isna().all() else 0
        consecutive_duplicates_raw[col] = max_consecutive
print("\nMax consecutive duplicates in raw data:")
for col, count in consecutive_duplicates_raw.items():
    print(f"{col}: {count} days")

print("Processed data shape (VNINDEX_iqr_filled_HSG_Adjust.csv):", df_processed.shape)
print("\nMissing values in Processed data(VNINDEX_iqr_filled_HSG_Adjust.csv):\n", df_processed.isna().sum())

# 4. Kiểm tra consecutive duplicates trong 
consecutive_duplicates = {}
for col in df_processed.columns:
    max_consecutive = (df_processed[col].diff() == 0).cumsum().value_counts().max()
    consecutive_duplicates[col] = max_consecutive
print("\nMax consecutive duplicates after processing:")
for col, count in consecutive_duplicates.items():
    print(f"{col}: {count} days")

# 5. Lưu dữ liệu
df_processed.to_csv('VNINDEX_cleaned_final.csv')
print("\nData saved to 'VNINDEX_cleaned_final.csv'")
files.download('VNINDEX_cleaned_final.csv')