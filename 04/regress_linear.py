import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 加载数据
df = pd.read_excel('./04/regressionobject.xlsx')  # 使用你的文件路径

# 准备特征和目标
X = df.iloc[:, 1:].values  # 假设前几列是特征
y = df.iloc[:, 0].values   # 假设第一列是目标

# 划分训练集（70%）和临时集（30%）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# 从临时集中划分验证集（15%）和测试集（15%）
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 定义线性回归模型
linear_regressor = LinearRegression()

# 训练模型
linear_regressor.fit(X_train, y_train)

# 对验证集进行预测
y_val_pred = linear_regressor.predict(X_val)

# 对测试集进行预测
y_test_pred = linear_regressor.predict(X_test)

# 计算验证集的评价指标
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

# 计算测试集的评价指标
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 输出验证集和测试集的结果
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation MAE: {val_mae:.4f}")
print(f"Validation R²: {val_r2:.4f}")

print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R²: {test_r2:.4f}")

# 输出回归系数（每个特征的权重）
coefficients = linear_regressor.coef_
feature_names = df.columns[1:]  # 获取特征名称（假设第1列是目标，后续是特征）
coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

# 输出每个特征的系数
print("\nFeature Coefficients:")
print(coeff_df)

