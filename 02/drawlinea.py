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
k1new=1.8711747562814567
alpha1new=1.438546907546309
beta1new=2.4324496558576656
cnew=-0.005416720500894235

def steinmetznew(f, Bm, k1, alpha1, belta1, c, T):
    return k1 * (f ** alpha1) * (Bm ** belta1) * (1 + c * T)

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
plt.scatter(predicted_values_new, target_data, color='purple', label='线性偏置模型', alpha=0.6)



# 绘制虚线
plt.plot([min(target_data), max(target_data)], 
         [min(target_data), max(target_data)], 
         color='red', linestyle='--', label='理论情况')

# 设置标签和标题
plt.xlabel('线性偏置预测值')
plt.ylabel('目标值')
plt.title('线性偏置预测值与目标值对比')
plt.legend()
plt.grid(True)
plt.savefig('./02/线性模型模型对比.png')
plt.show()
