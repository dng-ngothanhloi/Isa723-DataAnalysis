### 3.1.2.2 process_vnindex_IRQ_complete.py - tại bước Kiểm định sau cùng
### Bổ sung dùng HSG_log thay cho HSG & lưu vào cột HSG_Cleaned cho các xử lý tiếp theo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import files for downloading in Colab
from google.colab import files
# Đọc dữ liệu từ file CSV
try:
    # Attempt to read the file, parsing the first column as dates
    # Adjust the date_format based on your actual CSV file's date format
    # Common formats: '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'
    # The format in the original code '%mm/%dd/%yyyy' is likely incorrect.
    # Let's assume a common format like '%m/%d/%Y' or '%d/%m/%Y' first.
    # If your dates are like '01/01/2023', use '%m/%d/%Y'.
    # If your dates are like '01/Jan/2023', you might not need a format, pandas can often infer.
    # **IMPORTANT**: You need to adjust 'date_format' below to match your CSV file's date format.
    df = pd.read_csv('VNINDEX.csv')

    # Assuming the first column is the date column. Adjust index [0] if not.
    date_column_name = df.columns[0]

    # Convert the date column to datetime objects
    # Use the correct format string based on your file
    # Example: df[date_column_name] = pd.to_datetime(df[date_column_name], format='%m/%d/%Y', errors='coerce')
    # If the format is unknown or variable, let pandas infer it:
    df[date_column_name] = pd.to_datetime(df[date_column_name], errors='coerce')


    # Check if date parsing was successful
    if df[date_column_name].isna().all():
        print(f"Lỗi: Không thể parse cột '{date_column_name}' thành định dạng ngày. Vui lòng kiểm tra định dạng ngày trong file CSV.")
        df = pd.DataFrame() # Create an empty DataFrame
    else:
        # Set the date column as the index
        df.set_index(date_column_name, inplace=True)
        # Drop the original date column if it wasn't used as index
        if date_column_name in df.columns:
            df.drop(columns=[date_column_name], inplace=True)


except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'VNINDEX.csv'. Vui lòng cung cấp đường dẫn chính xác hoặc tải file lên.")
    df = pd.DataFrame() # Tạo DataFrame rỗng để tránh lỗi NameError sau này
except Exception as e:
    print(f"Đã xảy ra lỗi khi đọc hoặc xử lý file CSV: {e}")
    df = pd.DataFrame() # Tạo DataFrame rỗng

# --- Handle empty DataFrame early ---
if df.empty:
    print("Không có dữ liệu hợp lệ được đọc. Không thể tiến hành xử lý.")
    # You might want to exit or just skip the rest of the processing
    # exit(1) # Uncomment to exit
    # If not exiting, the rest of the code needs to handle an empty df
    pass # Continue but with an empty df

# Đặt cột ngày đã chuẩn hóa làm index
#df.set_index(date_column_name, inplace=True)

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
# Bổ sung dùng HSG_log thay cho HSG & lưu vào cột HSG_Cleaned cho các xử lý tiếp theo
df['HSG_Cleaned'] = np.log1p(df['HSG'])

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
# Ensure cleaned_columns is not empty before creating df_cleaned
if cleaned_columns:
    df_cleaned = df[cleaned_columns].copy()
    # Đổi tên cột _Cleaned về tên gốc
    df_cleaned.columns = [col.replace('_Cleaned', '') for col in df_cleaned.columns]
else:
    print("Không tìm thấy cột _Cleaned nào. Không thể tạo DataFrame cleaned.")
    df_cleaned = pd.DataFrame() # Create an empty DataFrame

# Bước 3: Tạo index thời gian đầy đủ
# Use the min and max dates from the cleaned data if available,
# otherwise use default dates. This makes the range dynamic.
start_date = df_cleaned.index.min() if not df_cleaned.empty else '03/01/2012'
end_date = df_cleaned.index.max() if not df_cleaned.empty else '05/26/2017'
print("\n Xử lý lại mảng dữ liệu ngày đầy đủ...")
print(f"Ngày bắt đầu: {start_date}")
print(f"Ngày kết thúc: {end_date}") 

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

## ----- Start thêm dữ liệu thiếu

def find_nan_streaks(series):
    is_nan = series.isna()
    streaks = []
    start = None
    for i, (idx, val) in enumerate(is_nan.items()):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            end = is_nan.index[i-1]
            length = (end - start).days + 1
            # Only include indices within the streak (from start to end)
            streak_indices = series.index[(series.index >= start) & (series.index <= end)]
            streaks.append((start, end, length, list(streak_indices)))
            start = None
    if start is not None:
        end = is_nan.index[-1]
        length = (end - start).days + 1
        # Only include indices within the streak (from start to end)
        streak_indices = series.index[(series.index >= start) & (series.index <= end)]
        streaks.append((start, end, length, list(streak_indices)))
    return streaks

# Step 4: Process missing data
def process_missing_data(df):
    df_processed = df.copy()    
    print("\n--- Starting missing data processing ---")
    for col in df_processed.columns:
        print(f"Processing column: {col}")
        series = df_processed[col]
        is_nan = series.isna()

        if not is_nan.any():
            print(f"  Column '{col}' has no missing values. Skipping.")
            continue

        # Find NaN streaks
        nan_streaks = find_nan_streaks(series)
        if not nan_streaks:
            print(f"  No consecutive missing data found in column '{col}'.")
            continue

        # Process each streak
        for start_date, end_date, length, streak_indices in nan_streaks:
                      
            if length <= 2:
                # Apply 5-day moving average for short streaks (e.g., weekends)
                filled_values = {}
                for current_nan_date in streak_indices:
                    # Find the 5 most recent *consecutive* valid indices before current_nan_date
                    valid_indices_before = series.index[
                        (series.index < current_nan_date) & (~series.isna())
                    ].sort_values(ascending=False)
                    
                    # Ensure consecutive valid indices
                    consecutive_indices = []
                    for idx in valid_indices_before:
                        if not consecutive_indices:
                            consecutive_indices.append(idx)
                        else:
                            if (consecutive_indices[-1] - idx).days == 1:
                                consecutive_indices.append(idx)
                            else:
                                break
                        if len(consecutive_indices) == 5:
                            break

                    if len(consecutive_indices) == 5:
                        values_for_mean = series.loc[consecutive_indices]
                        calculated_mean = values_for_mean.mean()
                        filled_values[current_nan_date] = calculated_mean
                    else:
                        # Use forward fill if not enough consecutive points
                        last_valid = series[series.index < current_nan_date][~series[series.index < current_nan_date].isna()].iloc[-1] if not series[series.index < current_nan_date][~series[series.index < current_nan_date].isna()].empty else np.nan
                        filled_values[current_nan_date] = last_valid

                df_processed.loc[streak_indices, col] = pd.Series(filled_values)
            
            else:
                # Apply forward fill (ffill) for long streaks (> 2 days)
                last_valid = series[series.index < start_date][~series[series.index < start_date].isna()].iloc[-1] if not series[series.index < start_date][~series[series.index < start_date].isna()].empty else np.nan
                df_processed.loc[streak_indices, col] = last_valid

    # Final pass: Handle remaining NaNs with ffill and bfill
    print("\n--- Applying final ffill and bfill for remaining NaNs ---")
    remaining_nans_before = df_processed.isna().sum().sum()
    print(f"NaN values before final ffill/bfill: {remaining_nans_before}")
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
    # Check for remaining NaNs
    missing_after_processing = df_processed.isna().sum()
    if missing_after_processing.any():
        print("\nRemaining NaN values after processing:")
        print(missing_after_processing[missing_after_processing > 0])
    else:
        print("\nNo NaN values remain after processing.")
    
    return df_processed
# Execute the processing
df_processed = process_missing_data(df_full)
##------ End Xử lý Thêm dữ liệu thiếu

# Vẽ biểu đồ 3: VNINDEX sau khi xử lý dữ liệu thiếu
if vnindex_col in df_processed.columns:
    plt.figure(figsize=(14, 4))
    ##df_processed['VNINDEX'].plot(title='VNINDEX After Missing Data Processing')
    plt.plot(df_processed[vnindex_col].index, df_processed[vnindex_col], label=f'{vnindex_col} After filled', alpha=0.8, color='red')
    plt.title(f'{vnindex_col} After Missing Data Processing')
    plt.xlabel('Date')
    plt.ylabel(vnindex_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{vnindex_col}_after_missing.png')
    plt.close()
    files.download(f'{vnindex_col}_after_missing.png')
    print(f"Đã lưu biểu đồ {vnindex_col} sau xử lý dữ liệu thiếu vào '{vnindex_col}_after_missing.png'")
else:
    print(f"Không tìm thấy cột '{vnindex_col}' trong dữ liệu đã xử lý để vẽ biểu đồ sau xử lý dữ liệu thiếu.")

# In mẫu dữ liệu sau xử lý
print("\nMẫu dữ liệu sau khi xử lý (5 dòng đầu):")
print(df_processed.head())

# Lưu File 2: Chỉ chứa cột dữ liệu đã xử lý, ghi đè tên cột gốc
output_filename_iqr_filled = 'VNINDEX_iqr_filled_HSG_Adjust.csv'
df_processed.to_csv(output_filename_iqr_filled)
print(f"\nĐã lưu dữ liệu (chỉ cột đã xử lý, tên cột gốc) vào '{output_filename_iqr_filled}'")
# Download the file in Colab
try:
    files.download(output_filename_iqr_filled)
except Exception as e:
    print(f"Không thể tự động tải xuống {output_filename_iqr_filled}. Vui lòng tải thủ công.")