import pandas as pd
import itertools
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot
import pingouin as pg
import scipy.stats as stats
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取数据
file_path = './03/Q3_data分析.xlsx'
excel_data = pd.ExcelFile(file_path)
df = excel_data.parse('Sheet1')

# 重命名列名，去除特殊字符和空格
df.rename(columns={'序号': 'id', '磁芯损耗，w/m3': '磁芯损耗'}, inplace=True)


# ANOVA方差分析
# 将温度、励磁波形、磁芯材料定义为分类变量
df['温度'] = df['温度'].astype('category')
df['励磁波形'] = df['励磁波形'].astype('category')
df['磁芯材料'] = df['磁芯材料'].astype('category')
df = df.drop(columns=['id'])  # 移除 id 列
df = df.drop(columns=['磁芯损耗'])  # 移除 id 列
mapping = {
    '温度': {'25度': 1, '50度': 2, '70度': 3, '90度': 4},
    '励磁波形': {'正弦波': 1, '三角波': 2, '梯形波': 3},
    '磁芯材料': {'材料1': 1, '材料2': 2, '材料3': 3, '材料4': 4}
}
df.replace(mapping, inplace=True)
# 计算 Spearman 相关系数矩阵
spearman_corr_matrix = df.corr(method='spearman')

# 打印 Spearman 相关系数矩阵
print(spearman_corr_matrix)

# 使用 seaborn 绘制热力图，使用蓝色渐变
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr_matrix, annot=True, cmap='Blues', fmt=".2f", vmin=-1, vmax=1)
plt.title('Spearman 相关系数矩阵')
plt.show()
