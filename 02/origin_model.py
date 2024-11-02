import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取用于回归数据
regdata = pd.read_excel('./02/regressiondata.xlsx')
def steinmetz(f,Bm,k1,alpha1,belta1):
    return k1*(f**alpha1)*(Bm**belta1)


# 提取用于拟合的数据
f_data = regdata['频率，Hz'].values
Bm_data = regdata['磁通密度峰值'].values
target_data = regdata['磁芯损耗，w/m3'].values

def steinmetz_curve_fit(params, f, Bm):
    return steinmetz(f, Bm, *params)

# 使用 curve_fit 进行拟合
# 初始参数猜测，例如 k1=1, alpha1=1, beta1=1
initial_guess = [1.0, 1.0, 1.0]
def steinmetz_combined(xdata, k1, alpha1, beta1):
    f, Bm = xdata
    return steinmetz(f, Bm, k1, alpha1, beta1)

# 将 f 和 Bm 组合成一个二维数组 (2, n)
xdata = np.vstack((f_data, Bm_data))

# 拟合曲线，返回最佳参数 popt
popt, pcov = curve_fit(steinmetz_combined, xdata, target_data, p0=initial_guess)

# 打印拟合结果
print(f"Fitted parameters: k1={popt[0]}, alpha1={popt[1]}, beta1={popt[2]}")

fitted_values = steinmetz(f_data, Bm_data, *popt)
#使用rMSE，mae，R2对拟合曲线进行评价
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
plt.vlines(sample_indices, 0, fitted_values[sample_indices], color='purple', linestyle='solid', label='原始模型拟合磁芯损耗')

# 设置标题和标签
plt.title('实际损耗与拟合损耗对比')
plt.xlabel('样本点')
plt.ylabel('磁芯损耗 (W/m^3)')

# 添加网格线
plt.grid(True)

# 添加图例
plt.legend()
plt.savefig('./02/原始模型拟合.png')

# 显示图像
plt.show()

