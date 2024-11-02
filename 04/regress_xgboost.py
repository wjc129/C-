import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_excel('./04/regressionobject.xlsx')  # 使用你的文件路径

# 准备特征和目标
X = df.iloc[:, 1:].values  # 假设第一列是目标，后面的列是特征
y = df.iloc[:, 0].values

# 数据标准化（可选，根据需要）
X_scaled = X  # 如果不需要标准化，可以直接使用原始数据

# 划分训练集（70%）、临时集（30%）
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 划分验证集（15%）和测试集（15%）
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 将数据转换为 DMatrix 格式（XGBoost 专用格式）
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# 定义 XGBoost 的参数
params = {
    'objective': 'reg:squarederror',  # 使用平方误差作为损失函数
    'eval_metric': 'rmse',            # 使用 RMSE 作为评价指标
    'max_depth': 6,                   # 树的最大深度
    'eta': 0.01,                      # 学习率
    'subsample': 0.9,                 # 子样本比例
    'colsample_bytree': 1,            # 每棵树使用的特征列比例
    'seed': 42                        # 固定随机种子，确保结果可复现
}

# 定义用于早停的评估集
evals = [(dtrain, 'train'), (dval, 'eval')]

# 记录每轮训练的结果
evals_result = {}

# 训练模型，使用早停，并保存每轮训练的结果
num_boost_round = 5000
early_stopping_rounds = 20

xg_reg = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    evals_result=evals_result,  # 保存训练过程中的结果
    verbose_eval=True  # 设置为 True 可以查看每轮的评估指标
)

# 输出最佳迭代轮数
print(f"Best iteration: {xg_reg.best_iteration}")

# 手动计算每个 epoch 的 MAE 和 R²
train_mae_list, val_mae_list = [], []
train_r2_list, val_r2_list = [], []

for epoch in range(xg_reg.best_iteration + 1):
    y_train_pred = xg_reg.predict(dtrain, iteration_range=(0, epoch+1))
    y_val_pred = xg_reg.predict(dval, iteration_range=(0, epoch+1))

    # 计算 MAE
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    train_mae_list.append(train_mae)
    val_mae_list.append(val_mae)

    # 计算 R²
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_r2_list.append(train_r2)
    val_r2_list.append(val_r2)

# 对训练集、验证集和测试集进行预测（最终模型），并取绝对值
y_train_pred = np.abs(xg_reg.predict(dtrain))  # 取绝对值
y_val_pred = np.abs(xg_reg.predict(dval))      # 取绝对值
y_test_pred = np.abs(xg_reg.predict(dtest))    # 取绝对值

# 计算训练集的评价指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# 计算验证集的评价指标
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

# 计算测试集的评价指标
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 输出训练集、验证集和测试集的结果
print(f"Training RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# 绘制 RMSE 变化曲线
epochs = len(evals_result['train']['rmse'])
plt.figure(figsize=(12, 6))
plt.plot(range(epochs), evals_result['train']['rmse'], label='Train RMSE')
plt.plot(range(epochs), evals_result['eval']['rmse'], label='Validation RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('XGBoost--RMSE Over Time')
plt.legend()
plt.grid(True)
plt.savefig('./04/rmse_over_time.png')
plt.show()

# 绘制 MAE 变化曲线
plt.figure(figsize=(12, 6))
plt.plot(range(len(train_mae_list)), train_mae_list, label='Train MAE')
plt.plot(range(len(val_mae_list)), val_mae_list, label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('XGBoost--MAE Over Time')
plt.legend()
plt.grid(True)
plt.savefig('./04/mae_over_time.png')
plt.show()

# 绘制 R² 变化曲线
plt.figure(figsize=(12, 6))
plt.plot(range(len(train_r2_list)), train_r2_list, label='Train R²')
plt.plot(range(len(val_r2_list)), val_r2_list, label='Validation R²')
plt.xlabel('Epochs')
plt.ylabel('R²')
plt.title('XGBoost--R² Over Time')
plt.legend()
plt.grid(True)
plt.savefig('./04/XGBoost--r2_over_time.png')
plt.show()

# 保存模型
xg_reg.save_model('./04/xgboost_model_with_abs.json')
