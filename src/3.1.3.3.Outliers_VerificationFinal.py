####3.1.1.3.Outliers_VerificationFinal.py
import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv('VNINDEX.csv', parse_dates=[0], index_col=0)

# Kiểm tra IQR với hệ số 3.0
for col in ['FLC', 'HSG', 'KDC', 'PPC']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[col][(df[col] < Q1 - 3.0*IQR) | (df[col] > Q3 + 3.0*IQR)]
    print(f"{col} - IQR (hệ số 3.0): {len(outliers)} ngoại lai ({len(outliers)/1342*100:.2f}%)")
# Kiểm tra tương quan
print("\nTương quan giữa các cột:")
print(df[['FLC', 'HSG', 'KDC', 'PPC']].corr())

# Biến đổi log cho HSG
df['HSG_log'] = np.log1p(df['HSG'])
print(f"\nSkewness HSG_log: {skew(df['HSG_log'].dropna()):.2f}")
Q1 = df['HSG_log'].quantile(0.25)
Q3 = df['HSG_log'].quantile(0.75)
IQR = Q3 - Q1
outliers = df['HSG_log'][(df['HSG_log'] < Q1 - 3.0*IQR) | (df['HSG_log'] > Q3 + 3.0*IQR)]
print(f"HSG_log - IQR (hệ số 3.0): {len(outliers)} ngoại lai ({len(outliers)/1342*100:.2f}%)")


# Vẽ so sánh HSG và KDC
plt.figure(figsize=(10, 6))
plt.plot(df['HSG'], label='HSG')
plt.plot(df['KDC'], label='KDC')
plt.title('So sánh HSG và KDC')
plt.legend()
plt.grid(True)
plt.savefig('hsg_kdc_compare.png')
plt.show()
plt.close()