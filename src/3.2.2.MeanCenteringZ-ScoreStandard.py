### Program: 3.2.2.MeanCenteringZ-ScoreStandard
####Mục tiêu:
##### * Mean centering: Đảm bảo ma trận hiệp phương sai chỉ phản ánh sự biến động, không bị lệch bởi trung bình (Giá trị ~0). 
#### * Z-score: Đảm bảo các cột (VNINDEX, HSG_log, BVH, ...) có thang đo đồng đều, tránh biến có phương sai lớn chi phối PCA => Biến ma trận hiệp phương sai thành ma trận tương quan, đưa các biến về cùng thang đorung bình của mỗi cột ~ 
####Độ lệch chuẩn của mỗi cột ~ 1.
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from google.colab import files

# 1. Đọc dữ liệu
df = pd.read_csv('VNINDEX_iqr_filled_HSG_Adjust.csv', parse_dates=[0], index_col=0)

# 2. Chuẩn hóa z-score (bao gồm mean centering)
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# 3.Lưu kết quả
df_standardized.to_csv('VNINDEX_standardized.csv')
files.download('VNINDEX_standardized.csv')
# 4. Kiểm tra trung bình và độ lệch chuẩn
print("Trung bình sau z-score:\n", df_standardized.mean())
print("\nĐộ lệch chuẩn sau z-score:\n", df_standardized.std())

##Out-putTrung bình sau z-score:
###VNINDEX      1.153596e-15
###BVH          2.883989e-17
###CSM         -4.037584e-16
###...
###HSG          5.767978e-16
###VNM         -5.767978e-17
###...
###CONGNGHE     2.883989e-16

###Độ lệch chuẩn sau z-score:
###VNINDEX      1.000254
###BVH          1.000254
###...
###HSG          1.000254
###...
###VN30         1.000254
###UPCOM        1.000254
###CONGNGHE     1.000254