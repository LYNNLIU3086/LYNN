import pandas as pd
import numpy as np

# ==============================
# 1. 读取 Excel 文件
# ==============================

input_file = "New-Spicydata.xlsx"
output_file = "cleaned_data.csv"

print("Reading Excel file...")
df = pd.read_excel(input_file, engine="openpyxl")

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())


# ==============================
# 2. 选择需要的列
# ==============================

required_columns = ['UTC', 'R_sat', 'Z_ring', 'RF score']

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df[required_columns].copy()


# ==============================
# 3. 删除空值
# ==============================

df = df.dropna()
print("After dropna:", df.shape)


# ==============================
# 4. 时间转换为 Unix timestamp
# ==============================

print("Converting UTC to Unix timestamp...")
df['UTC'] = pd.to_datetime(df['UTC'], errors='coerce')
df = df.dropna(subset=['UTC'])

df['timestamp'] = df['UTC'].astype(np.int64) // 10**9
df = df.drop(columns=['UTC'])


# ==============================
# 5. 数值归一化（避免Unity空间过大）
# ==============================

print("Normalizing R_sat and Z_ring...")

df['R_sat'] = df['R_sat'] / df['R_sat'].max()
df['Z_ring'] = df['Z_ring'] / df['Z_ring'].max()

# 可选：归一化 RF score
df['RF_score'] = df['RF score'] / df['RF score'].max()
df = df.drop(columns=['RF score'])


# ==============================
# 6. 生成 ID
# ==============================

df['id'] = range(len(df))


# ==============================
# 7. 选择 30 个“抵抗节点”
#    （RF_score 最高的30个）
# ==============================

print("Selecting resistance nodes...")
resistance_ids = df.nlargest(30, 'RF_score')['id']
df['isResistance'] = df['id'].isin(resistance_ids).astype(int)


# ==============================
# 8. 按时间排序
# ==============================

df = df.sort_values(by='timestamp')


# ==============================
# 9. 输出 CSV
# ==============================

df.to_csv(output_file, index=False)

print("Cleaned data saved as:", output_file)
print("Final shape:", df.shape)
print("Done.")