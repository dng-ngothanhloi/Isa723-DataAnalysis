### 3.1.2 process_vnindex_IRQ_complete.py
### Giải thuật fill-up dữ liệu thiết bị sai.
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import files for downloading in Colab
from google.colab import files

# Đọc dữ liệu từ file CSV
try:
    df = pd.read_csv('VNINDEX.csv', parse_dates=[0], index_col=0)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'VNINDEX.csv'. Vui lòng cung cấp đường dẫn chính xác hoặc tải file lên.")
    # Thoát script hoặc xử lý lỗi khác nếu file không tồn tại
    exit(1) # Thoát script

# --- Add this block to handle duplicate indices ---
# Kiểm tra và loại bỏ các dòng có index (ngày) bị trùng lặp
if not df.index.is_unique:
    print(f"Warning: Duplicate index entries found. Dropping duplicates, keeping the first.")
    df = df[~df.index.duplicated(keep='first')]
# --------------------------------------------------


# Vẽ biểu đồ 1: VNINDEX trước khi xử lý ngoại lai
# Use the column name directly if it exists after reading/cleaning
vnindex_col ='HSG'
if vnindex_col in df.columns:
    plt.figure(figsize=(14, 4))
    plt.plot(df[vnindex_col].index, df[vnindex_col], label=f'{vnindex_col}Original', alpha=0.8, color='blue')
    plt.title(f'{vnindex_col} Before Outlier Processing')
    plt.xlabel('Date')
    plt.ylabel(vnindex_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{vnindex_col}_after_outlier.png')
    plt.close()
    print(f'Đã lưu biểu đồ {vnindex_col} trước xử lý ngoại lai vào: {vnindex_col}_before_outlier.png')
else:
    print(f'Không tìm thấy cột {vnindex_col} để vẽ biểu đồ trước xử lý ngoại lai.')

# Bước 1: Xử lý ngoại lai bằng IQR
##•	Kiểm định bổ sung với IQR hệ số 1.5 ~ 3.0 đồng thời xem mối tương quan các mã để xem có mã nào không mang giá trị độc lập, đồng thời kiểm Skewness có đối tượng nghi ngờ HSG. Với phân tích kiểm định ngoại lai (Appendix 1) có thể kết luận:
##    -	Tất cả các cột (FLC, HSG, KDC, PPC) đều có ý nghĩa tài chính, đại diện cho các ngành quan trọng (bất động sản, thép, thực phẩm, điện).
##    -	FLC (2.46%), KDC (0%), PPC (0%) có tỷ lệ ngoại lai thấp hoặc không có, rất ổn định.
##    -	HSG có tỷ lệ ngoại lai giảm từ 27.58% (IQR 1.5) xuống 12.07% (IQR 3.0), và HSG_log chỉ còn 4.99% ngoại lai với skewness cải thiện (-0.40).
## • Ở trên đã xác định giữ lại tất cả các biến số IQR (hệ số 2.5 hoăc 3.0) nên sẽ áp dụng hệ số này cho dữ liệu mô hình tính toán, mục đich nhằm giảm tác động dữ liệu bị thay đổi bởi nội suy tuyến tính ở các điểm ngoại lai.
iRQRate=3.0
def remove_outliers_iqr(df, column):
    # Check if the column is numeric before processing outliers
    if pd.api.types.is_numeric_dtype(df[column]):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iRQRate * IQR ## change from 1.5 to 3.0
        upper_bound = Q3 + iRQRate * IQR
        # Tạo cột _is_outlier
        df[f'{column}_is_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
        # Tạo cột _Cleaned
        df[f'{column}_Cleaned'] = df[column].copy()
        # Thay thế ngoại lai bằng NaN
        df.loc[df[f'{column}_is_outlier'], f'{column}_Cleaned'] = np.nan
        # Nội suy tuyến tính
        df[f'{column}_Cleaned'] = df[f'{column}_Cleaned'].interpolate(method='linear')
        # Điền NaN ở đầu/cuối bằng ffill và bfill
        df[f'{column}_Cleaned'] = df[f'{column}_Cleaned'].fillna(method='ffill').fillna(method='bfill')
    else:
        print(f"Skipping outlier detection for non-numeric column: {column}")
        # Keep non-numeric columns as they are in _Cleaned and set _is_outlier to False
        df[f'{column}_Cleaned'] = df[column].copy()
        df[f'{column}_is_outlier'] = False

    return df

# Áp dụng IQR cho tất cả các cột số
# Process only the columns that are numeric and exist after duplicate removal
numerical_cols_after_dedup = df.select_dtypes(include=np.number).columns.tolist()
for col in numerical_cols_after_dedup:
    df = remove_outliers_iqr(df, col)

# Kiểm tra NaN sau xử lý ngoại lai (chỉ với các cột _Cleaned)
cleaned_columns_check_nan = [col for col in df.columns if col.endswith('_Cleaned')]
nan_counts_after_iqr = df[cleaned_columns_check_nan].isna().sum()
if not nan_counts_after_iqr[nan_counts_after_iqr > 0].empty:
    print("\nSố giá trị NaN trong mỗi cột _Cleaned sau khi xử lý ngoại lai:")
    print(nan_counts_after_iqr[nan_counts_after_iqr > 0])
else:
    print("\nKhông có giá trị NaN nào trong các cột _Cleaned sau khi xử lý ngoại lai.")


# Đếm số ngoại lai (chỉ với các cột số gốc đã xử lý)
print("\n--- Số điểm ngoại lai theo IQR ---")
for col in numerical_cols_after_dedup:
    outlier_flag_col = f'{col}_is_outlier'
    if outlier_flag_col in df.columns:
        outlier_count = df[outlier_flag_col].sum()
        print(f"Số điểm ngoại lai trong cột {col}: {outlier_count}")
    # else: This case should not happen if col is in numerical_cols_after_dedup

# Vẽ biểu đồ 2: VNINDEX sau khi xử lý ngoại lai
vnindex_cleaned_col = 'HSG_Cleaned'
vnindex_col = 'HSG'
if vnindex_cleaned_col in df.columns:
    plt.figure(figsize=(14, 4))
##    df[vnindex_cleaned_col].plot(title='VNINDEX After Outlier Processing (IQR)')
    plt.plot(df[vnindex_cleaned_col].index, df[vnindex_cleaned_col], label=f'{vnindex_col}  After Clean', alpha=0.8, color='green')
    plt.title(f'{vnindex_col} After Outlier Processing (IQR)')
    plt.xlabel('Date')
    plt.ylabel(vnindex_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{vnindex_col}_after_outlier.png')
    plt.close()
    print(f'Đã lưu biểu đồ {vnindex_col} sau xử lý ngoại lai vào {vnindex_col}_after_outlier.png')
else:
     print(f'Không tìm thấy cột {vnindex_cleaned_col} để vẽ biểu đồ sau xử lý ngoại lai.')

# Lưu File 1: Chứa cột gốc, cột _Cleaned, cột _is_outlier
output_filename_iqr_cleaned = 'VNINDEX_iqr_cleaned.csv'
df.to_csv(output_filename_iqr_cleaned)
print(f"\nĐã lưu dữ liệu (cột gốc + cột đã xử lý) vào '{output_filename_iqr_cleaned}'")
# Download the file in Colab
try:
    files.download(output_filename_iqr_cleaned)
except Exception as e:
    print(f"Không thể tự động tải xuống {output_filename_iqr_cleaned}. Vui lòng tải thủ công.")


# Bước 2: Tạo DataFrame chỉ chứa cột _Cleaned
cleaned_columns = [col for col in df.columns if col.endswith('_Cleaned')]
df_cleaned = df[cleaned_columns].copy()
# Đổi tên cột _Cleaned về tên gốc
df_cleaned.columns = [col.replace('_Cleaned', '') for col in df_cleaned.columns]

# Bước 3: Tạo index thời gian đầy đủ
# Use the min and max dates from the cleaned data if available,
# otherwise use default dates. This makes the range dynamic.
start_date = df_cleaned.index.min() if not df_cleaned.empty else '2012-01-03'
end_date = df_cleaned.index.max() if not df_cleaned.empty else '2017-05-26'
full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
# Now reindex should work as df_cleaned index is unique
df_full = df_cleaned.reindex(full_dates)

# Kiểm tra số giá trị NaN trước khi xử lý dữ liệu thiếu
missing_days = df_full.isna().sum()
if not missing_days[missing_days > 0].empty:
    print("\nSố giá trị NaN trong mỗi cột trước khi xử lý dữ liệu thiếu:")
    print(missing_days[missing_days > 0])
else:
    print("\nKhông có giá trị NaN nào trong các cột sau khi reindex.")


# Bước 4: Xác định các khoảng NaN liên tiếp > 2 ngày
def find_long_missing_streaks(series, threshold=2):
    is_nan = series.isna()
    if is_nan.all(): # Handle the case where the whole series is NaN
        return pd.Series(False, index=series.index) # No long streaks if all are NaN (or handle as needed)
    nan_groups = (is_nan != is_nan.shift()).cumsum()
    nan_groups = nan_groups[is_nan]
    if nan_groups.empty: # Handle case with no NaN values
        return pd.Series(False, index=series.index)
    group_counts = nan_groups.value_counts()
    long_groups = group_counts[group_counts > threshold].index
    long_streak_mask = nan_groups.isin(long_groups)
    # Reindex the mask to the original series index to ensure correct alignment
    return long_streak_mask.reindex(series.index, fill_value=False)


# Bước 5: Xử lý dữ liệu bị thiếu
df_processed = df_full.copy()

for col in df_processed.columns:
    # Áp dụng trung bình trượt bậc 5 cho *tất cả* các giá trị,
    # bao gồm cả NaN ban đầu. Điều này sẽ tạo ra NaN cho các khoảng trống lớn.
    # Lưu ý: rolling.mean() sẽ trả về NaN nếu cửa sổ chứa NaN.
    # Sau đó chúng ta sẽ điền các NaN này.
    df_processed[col] = df_full[col].rolling(window=5, min_periods=1, center=False).mean()

    # Xác định các khoảng NaN liên tiếp > 2 ngày dựa trên df_full (sau reindex)
    # Cần xác định lại mask cho df_full để đảm bảo đồng bộ
    long_streak_mask = find_long_missing_streaks(df_full[col], threshold=2)

    # Áp dụng forward fill cho các khoảng > 2 ngày đã xác định
    # Chúng ta cần fillna(method='ffill') trên df_full trước khi chọn lọc
    if long_streak_mask.any():
         ffill_values = df_full[col].ffill()
         # Chỉ điền ffill vào những vị trí thuộc về khoảng NaN dài ban đầu
         # Lấy các giá trị ffilled tại các index của khoảng dài
         df_processed.loc[long_streak_mask, col] = ffill_values.loc[long_streak_mask.index[long_streak_mask]]


    # Dùng forward fill cuối cùng để xử lý NaN còn lại
    df_processed[col] = df_processed[col].ffill()

    # Dùng backward fill cuối cùng để xử lý bất kỳ NaN nào còn sót lại ở cuối chuỗi
    df_processed[col] = df_processed[col].bfill()


# Kiểm tra NaN sau xử lý
missing_after_processing = df_processed.isna().sum()
if not missing_after_processing[missing_after_processing > 0].empty:
    print("\nSố giá trị NaN trong mỗi cột sau khi xử lý dữ liệu thiếu:")
    print(missing_after_processing[missing_after_processing > 0])
else:
    print("\nKhông có giá trị NaN nào trong các cột sau khi xử lý dữ liệu thiếu.")


# Vẽ biểu đồ 3: VNINDEX sau khi xử lý dữ liệu thiếu
if vnindex_cleaned_col in df_processed.columns:
    plt.figure(figsize=(14, 4))
    ##df_processed['VNINDEX'].plot(title='VNINDEX After Missing Data Processing')
    plt.plot(df_processed[vnindex_cleaned_col].index, df_processed[vnindex_cleaned_col], label=f'{vnindex_cleaned_col} After filled', alpha=0.8, color='red')
    plt.title(f'{vnindex_cleaned_col} After Missing Data Processing')
    plt.xlabel('Date')
    plt.ylabel(vnindex_cleaned_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{vnindex_cleaned_col}_after_missing.png')
    plt.close()
    print(f"Đã lưu biểu đồ {vnindex_cleaned_col} sau xử lý dữ liệu thiếu vào '{vnindex_cleaned_col}_after_missing.png'")
else:
    print(f"Không tìm thấy cột '{vnindex_cleaned_col}' trong dữ liệu đã xử lý để vẽ biểu đồ sau xử lý dữ liệu thiếu.")


# In mẫu dữ liệu sau xử lý
print("\nMẫu dữ liệu sau khi xử lý (5 dòng đầu):")
print(df_processed.head())

# Lưu File 2: Chỉ chứa cột dữ liệu đã xử lý, ghi đè tên cột gốc
output_filename_iqr_filled = 'VNINDEX_iqr_filled.csv'
df_processed.to_csv(output_filename_iqr_filled)
print(f"\nĐã lưu dữ liệu (chỉ cột đã xử lý, tên cột gốc) vào '{output_filename_iqr_filled}'")
# Download the file in Colab
try:
    files.download(output_filename_iqr_filled)
except Exception as e:
    print(f"Không thể tự động tải xuống {output_filename_iqr_filled}. Vui lòng tải thủ công.")