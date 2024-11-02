import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
file_path = './01_2/classfyobject.xlsx'
df = pd.read_excel(file_path)

# 创建一个用于保存图像的文件夹（如果不存在）
output_dir = './01_2/boxplots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置子图的行数和列数
num_columns = len(df.columns)
cols = 3  # 每行显示3个图
rows = 3

# 创建一个新图
plt.figure(figsize=(15, 5 * rows))

   


# 对每一列数据绘制箱型图
j=1
for i, column in enumerate(df.columns):
    if column in ['std', 'MaxVal', 'P2PVal', 'MinVal', 'skewness','kurtosis', 'spectral', 'harmonic', 'spectralentropy']:
        plt.subplot(rows, cols, j)  # 创建子图
        sns.boxplot(y=df[column], color='lightblue')
        plt.title(f"Boxplot of {column}")
        plt.ylabel(column)
        j=j+1

# 调整布局
plt.tight_layout()

# 保存图像到指定目录
output_path = os.path.join(output_dir, "combined_boxplots.png")
plt.savefig(output_path)  # 保存图像
plt.close()  # 关闭图像窗口，避免内存问题
