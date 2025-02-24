import matplotlib.pyplot as plt
import numpy as np

# 示例数据
categories = ["A", "B", "C", "D"]
values1 = [3, 2, 5, 7]
errors1 = [0.5, 0.4, 0.6, 0.7]

values2 = [4, 3, 6, 8]
errors2 = [0.6, 0.3, 0.5, 0.6]

x = np.arange(len(categories))  # 栏目位置

fig, ax = plt.subplots()

# 绘制第一个数据的误差棒
ax.errorbar(x, values1, yerr=errors1, fmt="o", label="shu1", capsize=5)

# 绘制第二个数据的误差棒
ax.errorbar(x, values2, yerr=errors2, fmt="o", label="shu2", capsize=5)

# 设置x轴刻度为类别名
ax.set_xticks(x)
ax.set_xticklabels(categories)

# 添加网格
ax.grid(True)

# 添加图例
ax.legend()

# 设置标题和轴标签
ax.set_title("non")
ax.set_xlabel("label")
ax.set_ylabel("val")

# 显示图表
plt.show()
