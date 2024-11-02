import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 加载数据
df = pd.read_excel('./04/regressionobject.xlsx')  # 使用你的文件路径

# 准备特征和目标
X = df.iloc[:, 1:].values  # 假设前几列是特征
y = df.iloc[:, 0].values   # 假设第一列是目标

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义随机森林回归模型
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_regressor.fit(X_train, y_train)

# 对测试集进行预测
y_pred = rf_regressor.predict(X_test)

# 计算评价指标
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出回归结果评价指标
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

# 输出特征重要性
importances = rf_regressor.feature_importances_
feature_names = df.columns[1:]  # 获取特征名称（假设第1列是目标，后续是特征）
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 输出每个特征的重要性
print("\nFeature Importances:")
print(importance_df)

# 如果需要保存特征重要性结果
importance_df.to_excel('./04/feature_importances.xlsx', index=False)
print("Feature importances have been saved to './04/feature_importances.xlsx'.")
