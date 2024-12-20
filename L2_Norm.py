import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成两个随机向量
a = np.random.rand(3)
b = np.random.rand(3)

# 计算欧几里得范数
L2 = np.linalg.norm(a - b)

# 使用散点图可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a[0], a[1], a[2], c='r', label='a')
ax.scatter(b[0], b[1], b[2], c='b', label='b')
ax.legend()
plt.show()