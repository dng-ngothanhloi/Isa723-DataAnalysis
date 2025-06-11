###3.1.1.1.Outliers_IQR_VNIndex.py
import pandas as pd
import numpy as np
from tabulate import tabulate
from google.colab import files

# Đọc dữ liệu (thay đổi đường dẫn tới file của bạn)
try:
    df = pd.read_csv('VNINDEX.csv', index_col=0, parse_dates=True)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file '6.VNINDEX.xlsx'. Vui lòng cung cấp đường dẫn chính xác.")
    exit(1)

# Hàm phát hiện và thay thế ngoại lai bằng IQR
##•	Kiểm định bổ sung với IQR hệ số 1.5 ~ 3.0 đồng thời xem mối tương quan các mã để xem có mã nào không mang giá trị độc lập,
##    đồng thời kiểm Skewness có đối tượng nghi ngờ HSG. Với phân tích kiểm định ngoại lai (Appendix 1) có thể kết luận:
##    -	Tất cả các cột (FLC, HSG, KDC, PPC) đều có ý nghĩa tài chính, đại diện cho các ngành quan trọng (bất động sản, thép, thực phẩm, điện).
##    -	FLC (2.46%), KDC (0%), PPC (0%) có tỷ lệ ngoại lai thấp hoặc không có, rất ổn định.
##    -	HSG có tỷ lệ ngoại lai giảm từ 27.58% (IQR 1.5) xuống 12.07% (IQR 3.0), và HSG_log chỉ còn 4.99% ngoại lai với skewness cải thiện (-0.40).
## • Ở trên đã xác định giữ lại tất cả các biến số IQR (hệ số 2.5 hoăc 3.0) nên sẽ áp dụng hệ số này cho dữ liệu mô hình tính toán, 
## mục đich nhằm giảm tác động dữ liệu bị thay đổi bởi nội suy tuyến tính ở các điểm ngoại lai.
iRQRate =3.0
def detect_and_replace_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iRQRate * IQR # change 1.5 => 3.0
    upper_bound = Q3 + iRQRate * IQR  # change 1.5 => 3.0
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    # Tạo bản sao để thay thế ngoại lai
    series_cleaned = series.copy()
    # Đánh dấu ngoại lai là NaN để nội suy
    series_cleaned[series < lower_bound] = np.nan
    series_cleaned[series > upper_bound] = np.nan
    # Nội suy tuyến tính cho các giá trị NaN
    series_cleaned = series_cleaned.interpolate(method='linear')
    return outliers, series_cleaned, lower_bound, upper_bound

# Phân tích và xử lý ngoại lai cho tất cả các cột số
outlier_summary = {}
total_rows = len(df)
df_cleaned = df.copy()  # Bản sao để lưu dữ liệu đã xử lý

for col in df.columns:
    outliers, cleaned_series, lower_bound, upper_bound = detect_and_replace_outliers_iqr(df[col])
    num_outliers = len(outliers)
    outlier_percentage = (num_outliers / total_rows) * 100 if total_rows > 0 else 0

    # Cập nhật cột đã xử lý
    df_cleaned[col] = cleaned_series

    # Lưu thông tin
    outlier_summary[col] = {
        'num_outliers': num_outliers,
        'outlier_percentage': outlier_percentage,
        'outliers': outliers
    }

# In chi tiết ngoại lai
print("PHÂN TÍCH VÀ XỬ LÝ DỮ LIỆU NGOẠI LAI BẰNG PHƯƠNG PHÁP IQR")
print("=" * 50)
for col, summary in outlier_summary.items():
    print(f"\nCột: {col}")
    print(f"  Số điểm ngoại lai: {summary['num_outliers']}")
    print(f"  Phần trăm ngoại lai: {summary['outlier_percentage']:.2f}%")
    if not summary['outliers'].empty:
        print("  Các điểm ngoại lai (ngày, giá trị):")
        for idx, value in summary['outliers'].items():
            print(f"    {idx.date()}: {value:.2f}")
    else:
        print("  Không có điểm ngoại lai.")

# Tạo bảng tổng kết phần trăm ngoại lai
summary_table = []
for col, summary in outlier_summary.items():
    recommendation = "Loại bỏ" if summary['outlier_percentage'] > 10 else "Giữ lại"
    summary_table.append({
        'Cột': col,
        'Số điểm ngoại lai': summary['num_outliers'],
        '% Ngoại lai': f"{summary['outlier_percentage']:.2f}%",
        'Khuyến nghị': recommendation
    })

# In bảng tổng kết
print("\nBẢNG TỔNG KẾT PHẦN TRĂM NGOẠI LAI")
print("=" * 50)
print(tabulate(summary_table, headers='keys', tablefmt='grid', stralign='center'))

# Lưu kết quả
# Lưu dữ liệu đã xử lý
df_cleaned.to_csv('VNINDEX_all_cols_cleaned(iqr)_outliers.csv')
#Download file
files.download('VNINDEX_all_cols_cleaned(iqr)_outliers.csv')
# Lưu báo cáo ngoại lai
outlier_report = []
for col, summary in outlier_summary.items():
    outlier_report.append({
        'Column': col,
        'Number_of_Outliers': summary['num_outliers'],
        'Outlier_Percentage': summary['outlier_percentage'],
        'Outlier_Details': summary['outliers'].to_dict(),
        'Recommendation': 'Remove' if summary['outlier_percentage'] > 10 else 'Keep'
    })
outlier_df = pd.DataFrame(outlier_report)
outlier_df.to_markdown('VNINDEX_outliers_iqr_summary.md', index=False)
outlier_df.to_excel('VNINDEX_outliers_iqr_summary.xlsx', index=False)
print("\nĐã lưu dữ liệu đã xử lý vào 'VNINDEX_cleaned.csv' và báo cáo tổng kết vào 'VNINDEX_outliers_iqr_summary.md'")
#Download file
files.download('VNINDEX_outliers_iqr_summary.md')
files.download('VNINDEX_outliers_iqr_summary.xlsx')