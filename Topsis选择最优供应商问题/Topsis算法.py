import numpy as np

#输入原始决策矩阵 (供应商 A, B, C, D)
#指标顺序：价格, 交货时间, 质量, 服务
data = (np.array
([
    [20, 10, 80, 75],  # A
    [22, 8, 85, 80],  # B
    [19, 12, 78, 70],  # C
    [21, 9, 82, 78]  # D
]))

#指标权重
weights = np.array([0.3, 0.2, 0.3, 0.2])

#指标类型：0 代表成本型（越小越好），1 代表效益型（越大越好）
kinds = [0, 0, 1, 1]

def topsis(data, weight, kind):
    # ---矩阵归一化 (向量归一化) ---
    # 计算每一列的平方和再开根号
    norm_data = data / np.sqrt(np.sum(data ** 2, axis=0))

    # ---带权重的归一化矩阵 ---
    weighted_data = norm_data * weight

    # ---确定正理想解和负理想解 ---
    best_v = []
    worst_v = []
    for i in range(len(kind)):
        if kind[i] == 1:  #效益型
            best_v.append(np.max(weighted_data[:, i]))
            worst_v.append(np.min(weighted_data[:, i]))
        else: #成本型
            best_v.append(np.min(weighted_data[:, i]))
            worst_v.append(np.max(weighted_data[:, i]))

    best_v = np.array(best_v)
    worst_v = np.array(worst_v)

    # --- 计算欧氏距离 ---
    # d_plus: 到最优解的距离, d_minus: 到最劣解的距离
    d_plus = np.sqrt(np.sum((weighted_data - best_v) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted_data - worst_v) ** 2, axis=1))

    # --- 计算综合得分 (C值) ---
    score = d_minus / (d_plus + d_minus)
    return score

#执行计算
scores = topsis(data, weights, kinds)

#打印结果
suppliers = ['供应商 A', '供应商 B', '供应商 C', '供应商 D']
results = zip(suppliers, scores)
sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

print("--- TOPSIS 评价结果 ---")
for name, score in sorted_results:
    print(f"{name}: 综合得分 = {score:.4f}")
print(f"\n最优供应商是: {sorted_results[0][0]}")