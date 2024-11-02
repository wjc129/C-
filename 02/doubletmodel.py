import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取用于回归数据
regdata = pd.read_excel('./02/regressiondata-t.xlsx')

# 定义 steinmetznew 函数
def steinmetznew(f, Bm, k1, alpha1, belta1, c, T):
    return k1 * (f ** alpha1) * (Bm ** belta1) * (1 + c * T*T)

# 提取用于拟合的数据
f_data = regdata['频率，Hz'].values
Bm_data = regdata['磁通密度峰值'].values
T_data = regdata['温度'].values
target_data = regdata['磁芯损耗，w/m3'].values

# 定义用于 curve_fit 的函数
def steinmetz_combined(xdata, k1, alpha1, beta1, c):
    f, Bm, T = xdata
    return steinmetznew(f, Bm, k1, alpha1, beta1, c, T)

# 将 f, Bm, T 组合成一个二维数组 (3, n)
xdata = np.vstack((f_data, Bm_data, T_data))

# 初始参数猜测
initial_guess = [1.0, 1.0, 1.0, 1.0]

# 拟合曲线，返回最佳参数 popt
popt, pcov = curve_fit(steinmetz_combined, xdata, target_data, p0=initial_guess)

# 打印拟合结果
print(f"Fitted parameters: k1={popt[0]}, alpha1={popt[1]}, beta1={popt[2]}, c={popt[3]}")

# 使用拟合的参数生成预测值
fitted_values = steinmetznew(f_data, Bm_data, popt[0], popt[1], popt[2], popt[3], T_data)

# 评价拟合曲线
mse = mean_squared_error(target_data, fitted_values)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

mae = mean_absolute_error(target_data, fitted_values)
print(f"Mean Absolute Error (MAE): {mae}")

r2 = r2_score(target_data, fitted_values)
print(f"R² Score: {r2}")

# 设定采样率，比如每5个点取1个
sampling_rate = 5
sample_indices = np.arange(0, len(f_data), sampling_rate)

# 绘制实际数据和拟合数据的竖线图
plt.figure(figsize=(10, 6))

# 绘制实际数据的小圆圈，使用 scatter
plt.scatter(sample_indices, target_data[sample_indices], facecolors='none', edgecolor='orange', label='实际磁芯损耗', s=10, marker='o')

# 绘制拟合数据的竖线，使用 vlines
plt.vlines(sample_indices, 0, fitted_values[sample_indices], color='purple', linestyle='solid', label='二次修正拟合磁芯损耗')

# 设置标题和标签
plt.title('实际损耗与拟合损耗对比')
plt.xlabel('样本点')
plt.ylabel('磁芯损耗 (W/m^3)')

# 添加网格线
plt.grid(True)

# 添加图例
plt.legend()
plt.savefig('./02/二次函数.png')
# 显示图像
plt.show()
