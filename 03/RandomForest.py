import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 读取数据
file_path = './03/Q3_data分析.xlsx'
excel_data = pd.ExcelFile(file_path)
df = excel_data.parse('Sheet1')

# 选择需要进行编码的列：温度、励磁波形、磁芯材料
columns_to_encode = ['温度', '励磁波形', '磁芯材料']
df.drop(columns=['序号'], inplace=True)  # 移除 id 列
# 使用 LabelEncoder 进行编码
label_encoders = {}
for column in columns_to_encode:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # 存储编码器，以便将来使用

# 将 '磁芯损耗，w/m3' 作为目标变量 y，其他作为自变量 X
X = df.drop(columns=['磁芯损耗，w/m3'])
y = df['磁芯损耗，w/m3']

# 随机森林模型训练
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
rf_importances = rf_model.feature_importances_

# 随机森林特征重要性
rf_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)

print("\n随机森林特征重要性：")
print(rf_feature_importance)
