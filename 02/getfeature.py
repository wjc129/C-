import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
rows_to_add = []
# 读取正弦波材料1数据
df_material1 = pd.read_excel('附件一（训练集）.xlsx',sheet_name='材料1')
for index, row in df_material1.iterrows():
    if row['励磁波形'] == '正弦波':  # 第3列（从0开始计数，所以是索引2）   
        rows_to_add.append(row)


sino = pd.DataFrame(rows_to_add, columns=df_material1.columns)
selected_columns = ['磁芯损耗，w/m3', '频率，Hz','磁通密度峰值']
max_B = np.zeros(sino.shape[0])
for index, row in sino.iterrows():
    max_B[index] = row[4:1028].max()

regressiondata = pd.DataFrame(columns=selected_columns)
regressiondata['磁芯损耗，w/m3']=sino['磁芯损耗，w/m3']
regressiondata['频率，Hz']=sino['频率，Hz']
regressiondata['磁通密度峰值']=max_B
regressiondata.to_excel('./02/regressiondata.xlsx',index=False)