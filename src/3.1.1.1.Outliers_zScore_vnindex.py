import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# ??c d? li?u (thay ???ng d?n b?ng ???ng d?n th?c t?)
df = pd.read_csv('VNINDEX.csv', index_col=0, parse_dates=True)

# X?c ??nh c?c c?t s?
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if 'is_outlier' in numerical_cols: # Remove if exists from previous runs
    numerical_cols.remove('is_outlier')
# Remove cleaned/outlier flag columns if they exist from previous runs
numerical_cols = [col for col in numerical_cols if not col.endswith('_cleaned') and not col.endswith('_is_outlier')]


# T?o m?t DataFrame ?? l?u tr? th?ng tin ngo?i lai cho t?t c? c?c c?t
outlier_info_df = pd.DataFrame(index=df.index)
outlier_counts = {} # Dictionary to store outlier counts per column

# ?p d?ng Z-score v? x? l? n?i suy cho t?ng c?t s?
for col in numerical_cols:
    print(f"\nProcessing column: {col}")

    # T?nh Z-score
    mean_col = df[col].mean()
    std_col = df[col].std()

    # Tr?nh chia cho 0 n?u ?? l?ch chu?n b?ng 0 (tr??ng h?p t?t c? gi? tr? gi?ng nhau)
    if std_col == 0:
        print(f"  Standard deviation for {col} is 0. No outliers detected by Z-score.")
        df[f'{col}_cleaned'] = df[col].copy()
        outlier_info_df[f'{col}_is_outlier'] = False # Mark all as not outlier
        outlier_counts[col] = 0
        continue # Move to the next column

    data_col = df[[col]].copy() # Use a temporary df for calculations
    data_col['z_score'] = (data_col[col] - mean_col) / std_col

    # Ph?t hi?n ngo?i lai (|Z| > 3)
    threshold = 3
    data_col['is_outlier'] = data_col['z_score'].abs() > threshold

    # L?u th?ng tin ngo?i lai
    outlier_info_df[f'{col}_is_outlier'] = data_col['is_outlier']

    # ??m s? l??ng ngo?i lai cho c?t n?y
    num_outliers = data_col['is_outlier'].sum()
    outlier_counts[col] = num_outliers

    # In c?c ?i?m ngo?i lai g?c cho c?t n?y
    outliers_col_data = data_col[data_col['is_outlier']][[col, 'z_score']]
    if not outliers_col_data.empty:
        print(f"  S? l??ng ngo?i lai trong c?t {col}: {num_outliers}")
        print(f"  C?c ?i?m ngo?i lai g?c trong c?t {col}:")
        print(outliers_col_data)
    else:
        print(f"  Kh?ng t?m th?y ngo?i lai trong c?t {col} theo ph??ng ph?p Z-score.")


    # X? l? ngo?i lai: Thay th? b?ng n?i suy tuy?n t?nh
    df[f'{col}_cleaned'] = df[col].copy()
    df.loc[data_col['is_outlier'], f'{col}_cleaned'] = np.nan
    df[f'{col}_cleaned'] = df[f'{col}_cleaned'].interpolate(method='linear')

    # In c?c ?i?m ngo?i lai sau khi x? l? n?i suy cho c?t n?y (optional, can be verbose)
    # if not outliers_col_data.empty:
    #     print(f"  C?c ?i?m ngo?i lai sau khi x? l? b?ng n?i suy tuy?n t?nh trong c?t {col}:")
    #     print(df[data_col['is_outlier']][f'{col}_cleaned'])


# K?t lu?n v? c?c c?t c? nhi?u ngo?i lai
print("\n--- T?m t?t s? l??ng ngo?i lai theo t?ng c?t ---")
for col, count in outlier_counts.items():
    print(f"C?t '{col}': {count} ngo?i lai")

print("\n--- C?c c?t c? th? c?n xem x?t k? h?n ---")
# B?n c? th? ??t m?t ng??ng ?? x?c ??nh 'nhi?u ngo?i lai'
# V? d?: C?t c? > 10 ?i?m ngo?i lai ho?c > 1% t?ng s? ?i?m d? li?u
total_data_points = len(df)
outlier_percentage_threshold = 0.01 # 1%
outlier_count_threshold = 10

for col, count in outlier_counts.items():
    percentage = (count / total_data_points) * 100 if total_data_points > 0 else 0
    if count > outlier_count_threshold or percentage > outlier_percentage_threshold:
        print(f"C?t '{col}': {count} ngo?i lai ({percentage:.2f}%) - C? th? c?n xem x?t k?.")

print("\n--- Ti?u ch? x?c ??nh ?i?m ngo?i lai c?n 'lo?i b?' ---")
print("Quy?t ??nh lo?i b? ho?n to?n m?t ?i?m ngo?i lai thay v? x? l? (nh? n?i suy) ph? thu?c v?o nhi?u y?u t?:")
print("1.  **Nguy?n nh?n g?y ra ngo?i lai:**")
print("    - N?u ngo?i lai l? do l?i nh?p li?u, l?i thi?t b?, ho?c c?c s? ki?n kh?ng l?p l?i v? kh?ng ph?n ?nh ??ng b?n ch?t d? li?u, vi?c lo?i b? c? th? l? h?p l?.")
print("    - N?u ngo?i lai l? m?t gi? tr? th?c t?, d? b?t th??ng (v? d?: m?t s? ki?n th? tr??ng l?n), vi?c lo?i b? c? th? l?m m?t th?ng tin quan tr?ng.")
print("2.  **M?c ti?u ph?n t?ch/m? h?nh:**")
print("    - M?t s? m? h?nh th?ng k? r?t nh?y c?m v?i ngo?i lai (v? d?: h?i quy tuy?n t?nh). ??i v?i c?c m? h?nh n?y, vi?c lo?i b? ho?c bi?n ??i ngo?i lai c? th? c?n thi?t.")
print("    - M?t s? m? h?nh kh?c ?t nh?y c?m h?n (v? d?: c?y quy?t ??nh, r?ng ng?u nhi?n).")
print("    - N?u m?c ti?u l? ph?t hi?n c?c s? ki?n b?t th??ng, ch?nh ngo?i lai l?i l? th?ng tin quan tr?ng c?n gi? l?i.")
print("3.  **S? l??ng ngo?i lai:**")
print("    - N?u ch? c? r?t ?t ?i?m ngo?i lai, vi?c lo?i b? ch?ng c? th? kh?ng ?nh h??ng ??ng k? ??n t?p d? li?u.")
print("    - N?u c? nhi?u ?i?m ngo?i lai, vi?c lo?i b? c? th? l?m gi?m k?ch th??c t?p d? li?u qu? nhi?u v? d?n ??n sai l?ch.")
print("4.  **T?m quan tr?ng c?a th?i gian:**")
print("    - Trong chu?i th?i gian, vi?c lo?i b? m?t ?i?m d? li?u t?o ra m?t 'kho?ng tr?ng' th?i gian c? th? ?nh h??ng ??n c?c ph?n t?ch ph? thu?c v?o t?nh li?n t?c c?a th?i gian.")
print("    - N?i suy (nh? linear interpolation ?? l?m) gi?p gi? l?i t?nh li?n t?c c?a chu?i th?i gian.")
print("5.  **Ki?n th?c chuy?n m?n (Domain Knowledge):**")
print("    - Hi?u bi?t v? l?nh v?c d? li?u ?ang l?m vi?c gi?p x?c ??nh xem m?t gi? tr? b?t th??ng c? ? ngh?a th?c t? hay kh?ng.")

print("\n**K?t lu?n:** Vi?c lo?i b? ho?n to?n ch? n?n ???c th?c hi?n khi b?n ch?c ch?n r?ng ?i?m ngo?i lai l? l?i d? li?u kh?ng th? kh?c ph?c v? kh?ng ch?a th?ng tin h?u ?ch cho m?c ti?u c?a b?n. Trong nhi?u tr??ng h?p, c?c ph??ng ph?p x? l? kh?c nh? bi?n ??i ho?c thay th? (n?i suy) l? ph? h?p h?n, ??c bi?t v?i d? li?u chu?i th?i gian ?? gi? l?i c?u tr?c th?i gian.")


# Optional: V? bi?u ?? cho m?t s? c?t sau x? l? (v? d?: VNINDEX v? m?t c?t kh?c)
# B?n c? th? t?y ch?nh ph?n n?y ?? hi?n th? c?c c?t b?n quan t?m
# (?? bao g?m ph?n v? bi?u ?? VNINDEX ? code tr??c, gi? l?i n?u mu?n)
# print("\nV? bi?u ?? cho c?t VNINDEX sau x? l?...")
# plt.figure(figsize=(12, 6))
# df['VNINDEX'].plot(label='VNINDEX g?c', alpha=0.5)
# df['VNINDEX_cleaned'].plot(label='VNINDEX sau x? l?', color='red')
# # Plot outliers for VNINDEX only for illustration
# # Ensure the outlier flag column exists before plotting
# if 'VNINDEX_is_outlier' in df.columns:
#     plt.scatter(df[df['VNINDEX_is_outlier']].index, df[df['VNINDEX_is_outlier']]['VNINDEX'],
#                 color='black', label='Ngo?i lai VNINDEX', marker='X', s=100)
# plt.title('Chu?i th?i gian VNINDEX v?i c?c ?i?m ngo?i lai (Z-score)')
# plt.legend()
# plt.savefig('VNINDEX_cleaned_comparison.png')
# plt.show()


# L?u k?t qu?
# Bao g?m c? c?t g?c, c?t x? l? v? c?t ??nh d?u ngo?i lai cho t?ng c?t
output_cols = []
for col in numerical_cols:
    output_cols.append(col)
    output_cols.append(f'{col}_cleaned')
    output_cols.append(f'{col}_is_outlier') # Add the outlier flag column

# K?t h?p outlier_info_df v?o df tr??c khi l?u
df = df.join(outlier_info_df)

output_filename = 'VNINDEX_all_cols_cleaned_outliers.csv'
# Ensure all output_cols actually exist in df after joining
existing_output_cols = [col for col in output_cols if col in df.columns]
df[existing_output_cols].to_csv(output_filename)

# Download the saved file
files.download(output_filename) 

