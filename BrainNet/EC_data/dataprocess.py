import re
import scipy.io as sio

# 输入文件路径
file_path = r"G:\XXX\EC_data\NC-A.txt"

# 用来存储结果
excite_idx = []   # 兴奋性
inhibit_idx = []  # 抑制性

# 正则表达式：匹配 "源脑区 23 -> 目标脑区 48 : 2.03460956 (兴奋性)"
pattern = re.compile(r"源脑区\s+(\d+)\s*->\s*目标脑区\s+(\d+)\s*:\s*([-+]?\d*\.?\d+)\s*\((兴奋性|抑制性)\)")

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            src = int(match.group(1)) + 1  # 源脑区 +1
            tgt = int(match.group(2)) + 1  # 目标脑区 +1
            # val = float(match.group(3))   # 数值（可以不用）
            conn_type = match.group(4)

            if conn_type == "兴奋性":
                excite_idx.append([src, tgt])
            else:
                inhibit_idx.append([src, tgt])

# 打印前几条结果
print("兴奋性连接示例：")
for i, conn in enumerate(excite_idx, 1):
    print(f"idx2{{1,{i}}} = {conn};")

print("\n抑制性连接示例：")
for i, conn in enumerate(inhibit_idx, 1):
    print(f"idx1{{1,{i}}} = {conn};")

# # 保存为 mat 文件，方便后续在 Matlab/Octave 里用
# sio.savemat("FLE整理结果.mat", {
#     "excite_idx": excite_idx,
#     "inhibit_idx": inhibit_idx
# })
