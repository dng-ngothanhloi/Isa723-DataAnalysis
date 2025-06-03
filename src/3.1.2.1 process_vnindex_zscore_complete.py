### 3.1.2 process_vnindex_zscorecomplete.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Đọc dữ liệu từ file CSV
try:
    df = pd.read_csv('VNINDEX.csv', parse_dates=[0], index_col=0)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'VNINDEX.csv'. Vui lòng cung cấp đường dẫn chính xác hoặc tải file lên.")
    df = pd.DataFrame() # Tạo DataFrame rỗng để tránh lỗi NameError sau này

# --- Added code to handle duplicate index entries ---
# Kiểm tra và loại bỏ các dòng có index (ngày) bị trùng lặp
if not df.index.is_unique:
    print(f"Warning: Duplicate index entries found. Dropping duplicates, keeping the first.")
    df = df[~df.index.duplicated(keep='first')]
# --------------------------------------------------

# Keep a copy of the original VNINDEX column before any processing for the chart
# Giữ lại một bản sao của cột VNINDEX gốc trước khi xử lý để vẽ biểu đồ so sánh
original_vnindex = pd.Series(dtype=float) # Khởi tạo rỗng
if 'VNINDEX' in df.columns:
    original_vnindex = df['VNINDEX'].copy()
else:
    print("Không tìm thấy cột 'VNINDEX' trong file gốc.")


# Bước 1: Xử lý ngoại lai bằng Z-score
def remove_outliers_zscore(df, column, threshold=3):
    # Check if the column is numeric before processing
    if pd.api.types.is_numeric_dtype(df[column]):
        # Tính Z-score
        # Tránh chia cho 0 nếu độ lệch chuẩn bằng 0
        if df[column].std() == 0:
            print(f"  Standard deviation for {column} is 0. No outliers detected by Z-score for this column.")
            df[f'{column}_is_outlier'] = False # Đánh dấu tất cả không phải ngoại lai
            df[f'{column}_Cleaned'] = df[column].copy()
        else:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            # Tạo cột _is_outlier
            df[f'{column}_is_outlier'] = z_scores > threshold
            # Tạo cột _Cleaned
            df[f'{column}_Cleaned'] = df[column].copy()
            # Thay thế ngoại lai bằng NaN
            df.loc[df[f'{column}_is_outlier'], f'{column}_Cleaned'] = np.nan

        # Nội suy tuyến tính - Áp dụng nội suy sau khi thay thế ngoại lai bằng NaN
        df[f'{column}_Cleaned'] = df[f'{column}_Cleaned'].interpolate(method='linear')
        # Điền NaN ở đầu/cuối bằng ffill (điền giá trị liền trước) và bfill (điền giá trị liền sau)
        df[f'{column}_Cleaned'] = df[f'{column}_Cleaned'].fillna(method='ffill').fillna(method='bfill')
    else:
        print(f"Skipping outlier detection for non-numeric column: {column}")
        # Giữ nguyên cột không phải số và tạo cột cleaned/outlier giả
        df[f'{column}_Cleaned'] = df[column].copy()
        df[f'{column}_is_outlier'] = False # Đánh dấu tất cả không phải ngoại lai

    return df

# Áp dụng Z-score cho tất cả các cột (chỉ xử lý các cột số)
original_columns = df.columns.tolist()
numerical_original_columns = df.select_dtypes(include=np.number).columns.tolist()

# Chỉ xử lý các cột ban đầu còn tồn tại sau khi loại bỏ dòng trùng lặp
cols_to_process = [col for col in original_columns if col in df.columns]

for col in cols_to_process:
    df = remove_outliers_zscore(df, col)


# Kiểm tra NaN sau xử lý ngoại lai (chỉ với các cột _Cleaned)
cleaned_columns_check_nan = [col for col in df.columns if col.endswith('_Cleaned')]
nan_counts_after_zscore = df[cleaned_columns_check_nan].isna().sum()
if not nan_counts_after_zscore[nan_counts_after_zscore > 0].empty:
    print("Số giá trị NaN trong mỗi cột _Cleaned sau khi xử lý ngoại lai:")
    print(nan_counts_after_zscore[nan_counts_after_zscore > 0])
else:
    print("Không có giá trị NaN nào trong các cột _Cleaned sau khi xử lý ngoại lai.")


# Đếm số ngoại lai (chỉ với các cột số gốc)
print("\n--- Số điểm ngoại lai theo Z-score ---")
for col in numerical_original_columns:
    outlier_flag_col = f'{col}_is_outlier'
    if outlier_flag_col in df.columns:
        outlier_count = df[outlier_flag_col].sum()
        print(f"Số điểm ngoại lai trong cột {col}: {outlier_count}")
    else:
        print(f"Cột cờ ngoại lai '{outlier_flag_col}' không tìm thấy (có thể do cột gốc không phải số).")


# Lưu File 1: Chứa cột gốc, cột _Cleaned, cột _is_outlier
output_filename_zscore = 'VNINDEX_zscore_cleaned.csv'
df.to_csv(output_filename_zscore)
print(f"\nĐã lưu dữ liệu (cột gốc + cột đã xử lý) vào '{output_filename_zscore}'")

try:
    files.download(output_filename_zscore)
except Exception as e:
    print(f"Không thể tự động tải xuống {output_filename_zscore}. Vui lòng tải thủ công.")


# Bước 2: Tạo DataFrame chỉ chứa cột _Cleaned
cleaned_columns_for_full_df = [col for col in df.columns if col.endswith('_Cleaned')]
df_cleaned = df[cleaned_columns_for_full_df].copy()
# Đổi tên cột _Cleaned về tên gốc
df_cleaned.columns = [col.replace('_Cleaned', '') for col in df_cleaned.columns]

# Bước 3: Tạo index thời gian đầy đủ
# Sử dụng ngày nhỏ nhất và lớn nhất từ index gốc nếu có dữ liệu, nếu không dùng ngày mặc định
start_date = df.index.min() if not df.empty else '2012-01-03'
end_date = df.index.max() if not df.empty else '2017-05-31'
full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
df_full = df_cleaned.reindex(full_dates)

# Kiểm tra số giá trị NaN trước khi xử lý dữ liệu thiếu
missing_days_before_processing = df_full.isna().sum()
if not missing_days_before_processing[missing_days_before_processing > 0].empty:
    print("\nSố giá trị NaN trong mỗi cột trước khi xử lý dữ liệu thiếu (sau reindex):")
    print(missing_days_before_processing[missing_days_before_processing > 0])
else:
    print("\nKhông có giá trị NaN nào trong các cột sau khi reindex.")

# Bước 4: Xác định các khoảng NaN liên tiếp > 2 ngày
def find_long_missing_streaks(series, threshold=2):
    is_nan = series.isna()
    if is_nan.all(): # Xử lý trường hợp toàn bộ series là NaN
        return is_nan
    nan_groups = (is_nan != is_nan.shift()).cumsum()
    nan_groups = nan_groups[is_nan]
    if nan_groups.empty: # Xử lý trường hợp không có NaN
        return pd.Series(False, index=series.index)
    group_counts = nan_groups.value_counts()
    long_groups = group_counts[group_counts > threshold].index
    long_streak_mask = nan_groups.isin(long_groups)
    return long_streak_mask.reindex(series.index, fill_value=False)


# Bước 5: Xử lý dữ liệu bị thiếu
df_processed = df_full.copy()

for col in df_processed.columns:
    # Áp dụng Rolling Mean cho các khoảng trống NGẮN (<= threshold)
    # Sử dụng cột từ df_full để rolling mean vì nó có index đầy đủ
    df_processed[col] = df_full[col].rolling(window=5, min_periods=1, center=False).mean()

    # Xác định các khoảng NaN liên tiếp DÀI (> threshold) dựa trên df_full gốc
    long_streak_mask_original = find_long_missing_streaks(df_full[col], threshold=2)

    # Áp dụng Forward Fill (ffill) cho các khoảng DÀI đã xác định
    if long_streak_mask_original.any():
        ffill_values = df_full[col].ffill()
        # Chỉ điền ffill vào những vị trí thuộc về khoảng NaN dài ban đầu
        df_processed.loc[long_streak_mask_original, col] = ffill_values.loc[long_streak_mask_original]

    # Dùng Forward Fill cuối cùng để xử lý bất kỳ NaN nào còn sót lại ở đầu chuỗi hoặc sau rolling/ffill
    df_processed[col] = df_processed[col].ffill()

    # Dùng Backward Fill (bfill) cuối cùng để xử lý bất kỳ NaN nào còn sót lại ở cuối chuỗi
    df_processed[col] = df_processed[col].bfill()


# Kiểm tra NaN sau xử lý dữ liệu thiếu
missing_after_processing = df_processed.isna().sum()
if not missing_after_processing[missing_after_processing > 0].empty:
    print("\nSố giá trị NaN trong mỗi cột sau khi xử lý dữ liệu thiếu:")
    print(missing_after_processing[missing_after_processing > 0])
else:
    print("\nKhông có giá trị NaN nào trong các cột sau khi xử lý dữ liệu thiếu.")


# In mẫu dữ liệu sau xử lý
print("\nMẫu dữ liệu sau khi xử lý (5 dòng đầu):")
print(df_processed.head())


# --- Add plotting for VNINDEX comparison as separate charts ---
if 'VNINDEX' in df_processed.columns:
    print("\nVẽ biểu đồ so sánh cột VNINDEX qua các bước xử lý (biểu đồ riêng)...")

    # Biểu đồ 1: VNINDEX Gốc
    if not original_vnindex.empty:
        plt.figure(figsize=(14, 4))
        plt.plot(original_vnindex.index, original_vnindex, label='VNINDEX Gốc', alpha=0.8, color='blue')
        plt.title('1. Chuỗi thời gian VNINDEX Gốc')
        plt.xlabel('Ngày')
        plt.ylabel('Giá trị VNINDEX')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Biểu đồ 2: VNINDEX sau Xử lý Ngoại lai (Z-score + Nội suy)
    vnindex_cleaned_col = 'VNINDEX_Cleaned'
    if vnindex_cleaned_col in df.columns:
        plt.figure(figsize=(14, 4))
        # Use the index from df, which contains the data after Z-score
        plt.plot(df.index, df[vnindex_cleaned_col], label='VNINDEX sau Xử lý Ngoại lai (Z-score + Nội suy)', color='orange')
        plt.title('2. Chuỗi thời gian VNINDEX sau Xử lý Ngoại lai')
        plt.xlabel('Ngày')
        plt.ylabel('Giá trị VNINDEX')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
         print(f"Không tìm thấy cột '{vnindex_cleaned_col}' để vẽ biểu đồ sau xử lý ngoại lai.")


    # Biểu đồ 3: VNINDEX sau Xử lý Dữ liệu Thiếu (Rolling Mean + ffill/bfill)
    plt.figure(figsize=(14, 4))
    # Use the index from df_processed, which has the full date range
    plt.plot(df_processed.index, df_processed['VNINDEX'], label='VNINDEX sau Xử lý Thiếu (Rolling Mean + ffill/bfill)', color='red')
    plt.title('3. Chuỗi thời gian VNINDEX sau Xử lý Dữ liệu Thiếu')
    plt.xlabel('Ngày')
    plt.ylabel('Giá trị VNINDEX')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Đã hiển thị 3 biểu đồ riêng biệt cho cột VNINDEX qua các bước xử lý.")

else:
    print("\nKhông tìm thấy cột 'VNINDEX' trong dữ liệu đã xử lý để vẽ biểu đồ.")
# ------------------------------------------


# Lưu File 2: Chỉ chứa cột dữ liệu đã xử lý, ghi đè tên cột gốc
output_filename_filled = 'VNINDEX_zscore_filled.csv'
df_processed.to_csv(output_filename_filled)
print(f"\nĐã lưu dữ liệu (chỉ cột đã xử lý, tên cột gốc) vào '{output_filename_filled}'")

try:
    files.download(output_filename_filled)
except Exception as e:
    print(f"Không thể tự động tải xuống {output_filename_filled}. Vui lòng tải thủ công.")