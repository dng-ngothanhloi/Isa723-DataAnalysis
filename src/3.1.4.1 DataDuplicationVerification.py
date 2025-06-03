###3.1.4.1 DataDuplicationVerification
import pandas as pd
import numpy as np
from google.colab import files

# 1. Đọc dữ liệu
df = pd.read_csv('VNINDEX_iqr_filled_HSG_Adjust.csv', parse_dates=[0], index_col=0)
# --- Added code to round numeric columns ---
# Chọn tất cả các cột có kiểu dữ liệu số
numeric_cols = df.select_dtypes(include=np.number).columns

# Áp dụng làm tròn 2 chữ số thập phân cho các cột số
df[numeric_cols] = df[numeric_cols].round(2)
print("Đã làm tròn các cột số đến 2 chữ số thập phân.")
# -------------------------------------------

# 2. Định lượng số lượng và tỷ lệ trùng lặp
total_rows = len(df)
duplicated_rows = df.index.duplicated().sum()
duplication_ratio = duplicated_rows / total_rows * 100

print(f"Total rows: {total_rows}")
print(f"Duplicated rows: {duplicated_rows}")
print(f"Duplication ratio (%): {duplication_ratio:.2f}%")

# 3. Xác định phân bố trùng lặp
duplicated_dates = df.index[df.index.duplicated()].unique()
print("\nDuplicated dates:", duplicated_dates[:10])  # In 10 ngày đầu tiên

# 4. Kiểm tra giá trị trùng lặp liên tiếp
# Tính số lần giá trị lặp lại liên tiếp trong mỗi cột
consecutive_duplicates = {}
for col in df.columns:
    # Kiểm tra giá trị giống nhau liên tiếp
    is_consecutive_same = (df[col].diff() == 0).cumsum()
    consecutive_counts = is_consecutive_same.value_counts()
    max_consecutive = consecutive_counts[consecutive_counts > 1].max() if not consecutive_counts[consecutive_counts > 1].empty else 0
    consecutive_duplicates[col] = max_consecutive

print("\nMax consecutive identical values per column:")
for col, count in consecutive_duplicates.items():
    print(f"{col}: {count} days")

# 5. Kiểm tra ngày trùng lặp có phải ngày không giao dịch
# Giả định ngày không giao dịch là thứ Bảy/Chủ nhật
df['weekday'] = df.index.weekday
non_trading_days = df[df['weekday'].isin([5, 6])]  # 5: Saturday, 6: Sunday
non_trading_duplicates = non_trading_days.index[non_trading_days.index.duplicated()]
print(f"\nNon-trading duplicated days: {len(non_trading_duplicates)}")

# 6. Lưu kết quả kiểm tra
with open('duplication_report.txt', 'w') as f:
    f.write(f"Total rows: {total_rows}\n")
    f.write(f"Duplicated rows: {duplicated_rows}\n")
    f.write(f"Duplication ratio: {duplication_ratio:.2f}%\n")
    f.write(f"Duplicated dates: {duplicated_dates[:10]}\n")
    f.write("Max consecutive duplicates:\n")
    for col, count in consecutive_duplicates.items():
        f.write(f"{col}: {count} days\n")
    f.write(f"Non-trading duplicated days: {len(non_trading_duplicates)}\n")
    f.write("End of report.")
    print("Kết quả kiểm tra đã được lưu vào 'duplication_report.txt'")
files.download('duplication_report.txt')    