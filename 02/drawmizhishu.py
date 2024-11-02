import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# 定义 Steinmetz 函数
k1 = 1.4997610634560803
alpha1 = 1.4296334584100037
beta1 = 2.471253706207208

def steinmetz(f, Bm, k1, alpha1, beta1):
    return k1 * (f ** alpha1) * (Bm ** beta1)

# 新的 Steinmetz 函数
k1new = 4.738384649803147
alpha1new = 1.4634839162196402
beta1new = 2.4485146685595067
cnew = -0.40175682182270506

def steinmetznew(f, Bm, k1new, alpha1new, beta1new, c, T):
    return k1new * (f ** alpha1new) * (Bm ** beta1new) * (T ** c)

# 读取数据
df = pd.read_excel('./02/regressiondata-t.xlsx')
# 提取数据
f_data = df['频率，Hz'].values
Bm_data = df['磁通密度峰值'].values
T_data = df['温度'].values
target_data = df['磁芯损耗，w/m3'].values

# 计算预测值
predicted_values_old = steinmetz(f_data, Bm_data, k1, alpha1, beta1)
predicted_values_new = steinmetznew(f_data, Bm_data, k1new, alpha1new, beta1new, cnew, T_data)

# 绘制散点图
plt.figure(figsize=(10, 6))

# 原始模型散点图
plt.scatter(predicted_values_old, target_data, color='orange', label='原始模型', alpha=0.6)

# 新模型散点图
plt.scatter(predicted_values_new, target_data, color='purple', label='幂指数修正模型', alpha=0.6)



# 绘制虚线
plt.plot([min(target_data), max(target_data)], 
         [min(target_data), max(target_data)], 
         color='red', linestyle='--', label='理论情况')

# 设置标签和标题
plt.xlabel('幂指数偏置预测值')
plt.ylabel('目标值')
plt.title('幂指数偏置预测值与目标值对比')
plt.legend()
plt.grid(True)
plt.savefig('./02/幂指数模型对比.png')
plt.show()
