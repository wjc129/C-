import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义 Steinmetz 函数
def steinmetz(f, Bm, k1, alpha1, beta1):
    return k1 * (f ** alpha1) * (Bm ** beta1)

# 给定参数
k1 = 1.4997610634560803
alpha1 = 1.4296334584100037
beta1 = 2.471253706207208

# 定义频率和 Bm 的范围
frequencies = np.linspace(50000, 500000, 100)  # 频率范围
Bm_values = np.linspace(0, 0.4, 100)            # Bm 范围

# 创建一个网格并计算对应的 Steinmetz 值
F, B = np.meshgrid(frequencies, Bm_values)
Z = steinmetz(F, B, k1, alpha1, beta1)

# 绘制三维曲面
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(F, B, Z, cmap='viridis')

# 添加颜色条
cbar = fig.colorbar(surf)
cbar.set_label('Steinmetz Output')

# 设置标签和标题
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Bm')
ax.set_zlabel('Steinmetz Output')
ax.set_title('3D Surface of Steinmetz Function')

# 保存图像
plt.savefig('./02/steinmetz_surface.png')

# 显示图像
plt.show()
