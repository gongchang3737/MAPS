# 格式检测
# import chardet

# with open('nodelist_utf8.csv', 'rb') as f:
#     raw = f.read()
# result = chardet.detect(raw)
# print(raw)
# print(result)  # 输出比如 {'encoding': 'utf-8', 'confidence': 0.99}

# 格式转换，其中有无BOM的区别在于：
# 用 UTF‑8 编码但不带 BOM（即使用 utf‑8 而非 utf‑8‑sig）保存 CSV 时，Excel（尤其在 Windows 上）
# 通常不会自动识别它是 UTF‑8，默认会按 ANSI 或本地编码打开，导致中文显示为乱码。
# 而如果文件带有 BOM（即用 utf‑8‑sig 编码），Excel 能正确识别编码，从而正常显示中文字符。
# 不过在python代码中都可以正常读取，仅在 Excel 中打开时有区别。
import pandas as pd

# 读取xlsx文件
df = pd.read_excel('nodelist.xlsx', dtype=str)

# 写出成 utf-8，无 BOM
df.to_csv('nodelist_utf8.csv', encoding='utf-8', index=False)

# 或者写出带 BOM 的文件
df.to_csv('nodelist_utf8_bom.csv', encoding='utf-8-sig', index=False)
data = pd.read_csv('nodelist_utf8.csv', encoding='utf-8')
data_bom = pd.read_csv('nodelist_utf8_bom.csv', encoding='utf-8-sig')
print(data)
print(data_bom)

# 读取xlsx文件
df = pd.read_excel('edgelist_detailed.xlsx', dtype=str)

# 写出成 utf-8，无 BOM
df.to_csv('edgelist_detailed_utf8.csv', encoding='utf-8', index=False)

# 或者写出带 BOM 的文件
df.to_csv('edgelist_detailed_utf8_bom.csv', encoding='utf-8-sig', index=False)
data = pd.read_csv('edgelist_detailed_utf8.csv', encoding='utf-8')
data_bom = pd.read_csv('edgelist_detailed_utf8_bom.csv', encoding='utf-8-sig')
print(data)
print(data_bom)
