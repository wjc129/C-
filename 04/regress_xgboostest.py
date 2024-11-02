import pandas as pd
import numpy as np
from pyswarm import pso
import xgboost as xgb
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
# 读取测试数据
df = pd.read_excel('./04/regressionobject_test.xlsx')  # 使用新的特征数据文件路径
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('./04/xgboost_model.json')
dnew = xgb.DMatrix(df.values)
result = np.abs(xgb_model.predict(df.values))
result_df = pd.DataFrame(result, columns=['Predicted_Value'])

# 将结果保存为 Excel 文件
output_path = './04/xgboostpredicted_results.xlsx'
result_df.to_excel(output_path, index=False)

# 确认文件保存成功
print(f"Predicted results successfully saved to {output_path}")
print(result)


# 提取预测值列
values = result_df['Predicted_Value'].values

# 绘制图形
plt.figure(figsize=(12, 6))
plt.bar(range(len(values)), values)  # 使用 bar 绘制每个采样点的竖线
plt.xlabel('Sample Index')  # X轴表示采样点的索引
plt.ylabel('Predicted Value')  # Y轴表示每个采样点的预测值
plt.title('Predicted Values for Each Sample (Vertical Lines)')
plt.grid(True)

# 显示图像
plt.show()