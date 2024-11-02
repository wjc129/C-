import pandas as pd
import numpy as np
from pyswarm import pso
import xgboost as xgb
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 读取训练数据
df = pd.read_excel('./05/regressionobject.xlsx')

# 计算传输磁能和磁芯损耗的最大最小值
transmag = df['频率'] * df['MaxVal']
max_transmag = transmag.max()
min_transmag = transmag.min()

cost = df['磁芯损耗']
max_cost = cost.max()
min_cost = cost.min()

freq = df['频率']
max_freq = freq.max()
min_freq = freq.min()

maxval = df['MaxVal']
max_maxval = maxval.max()
min_maxval = maxval.min()

# 加载训练好的 XGBoost 模型
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('./05/xgboost_model_with_abs.json')

# 用于存储结果
lambda1_values = []
optimal_parameters = []
P_loss_values = []
W_trans_values = []

# 定义目标函数生成器
def create_objective_function(lambda1):
    def objective_function(x):
        possible_materials = [1, 2, 3, 4]  
        possible_waveforms = [1, 2, 3]  
        possible_temperatures = [25, 50, 70, 90]  

        x[1] = min(possible_materials, key=lambda m: abs(m - x[1]))
        x[4] = min(possible_waveforms, key=lambda w: abs(w - x[4]))
        x[2] = min(possible_temperatures, key=lambda t: abs(t - x[2]))

        x_scal = np.array([x])  
        P_loss = np.abs(xgb_model.predict(x_scal)[0])  

        f = x[3]  
        Bm = x[0]  
        W_trans = f * Bm  

        normalized_P_loss = (P_loss - min_cost) / (max_cost - min_cost)
        normalized_W_trans = (W_trans - min_transmag) / (max_transmag - min_transmag)

        lambda2 = 1 - lambda1  
        objective_value = lambda1 * normalized_P_loss - lambda2 * normalized_W_trans

        return objective_value
    return objective_function

# 创建函数列表
lambda_functions = [create_objective_function(lambda1) for lambda1 in np.arange(0, 1.05, 0.05)]
lb = [min_maxval, 1, 25, min_freq, 1]  # [MaxVal, 材料, 温度, 频率, 励磁波形] 下边界
ub = [max_maxval, 4, 90, max_freq, 3]  # [MaxVal, 材料, 温度, 频率, 励磁波形] 上边界
# 优化并记录结果
for lambda1, func in zip(np.arange(0, 1.05, 0.05), lambda_functions):
    initial_x = [1, 1, 25, min_freq, 1]  
    xopt, fopt = pso(func, lb, ub, swarmsize=100, maxiter=100)
    
    
    optimal_values = xopt
    P_loss = np.abs(xgb_model.predict(np.array([optimal_values]).reshape(1, -1))[0])
    f_opt = optimal_values[3]  # 最优频率
    Bm_opt = optimal_values[0]  # 最优 MaxVal (磁通密度峰值)
    W_trans = f_opt * Bm_opt
    

    # 存储结果
    lambda1_values.append(lambda1)
    optimal_parameters.append(optimal_values)
    P_loss_values.append(P_loss)
    W_trans_values.append(W_trans)

    # 输出结果
    print(f'λ1: {lambda1:.2f}, 结果值:{fopt},优化参数: {optimal_values}, P_loss: {P_loss:.2f}, W_trans: {W_trans:.2f}')

# 绘制结果
plt.figure(figsize=(12, 6))

# 绘制 P_loss 和 W_trans 随 λ1 的变化
plt.plot(lambda1_values, P_loss_values, marker='o', label='磁芯损耗')
plt.plot(lambda1_values, W_trans_values, marker='o', label='传输磁能')
plt.title('磁芯损耗 和 传输磁能 随 λ1 变化')
plt.xlabel('λ1')
plt.ylabel('值')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('./05/losses_vs_lambda1.png')
plt.show()
