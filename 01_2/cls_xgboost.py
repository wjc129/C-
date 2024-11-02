import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
# 1. 数据预处理
start_time = time.time()  # 记录开始时间

df_ts = pd.read_excel('./01_2/classfyobject_test.xlsx')
features_ts = df_ts.iloc[:, list(range(1, 7)) + list(range(8, 11))].values

df = pd.read_excel('./01_2/classfyobject.xlsx')

# 打乱数据顺序
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

labels = df.iloc[:, 0].values  # 第一列为分类标签
features = df.iloc[:, list(range(2, 8)) + list(range(9, 12))].values

# 将标签编码为数值
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 对特征进行归一化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features_ts = scaler.transform(features_ts)  # 使用训练数据的scaler

# 将数据划分为训练集、验证集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(scaled_features, encoded_labels, test_size=0.3, random_state=42)  # 训练集占 70%
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 剩下的 30% 中再划分验证集和测试集各占 15%

end_preprocessing_time = time.time()  # 记录预处理结束时间
print(f"Data preprocessing time: {end_preprocessing_time - start_time:.2f} seconds")

# 2. 使用XGBoost进行训练
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)
dtest_ts = xgb.DMatrix(scaled_features_ts)

# 设置XGBoost参数
params = {
    'objective': 'multi:softmax',  # 多分类任务
    'num_class': len(np.unique(encoded_labels)),  # 类别数量
    'eval_metric': 'mlogloss',  # 多分类的对数损失
    'max_depth': 4,  # 树的最大深度
    'eta': 0.1,  # 学习率
    'subsample': 0.8,  # 训练样本的比例
    'colsample_bytree': 0.8  # 每棵树使用特征的比例
}

# 设置验证集进行训练
watchlist = [(dtrain, 'train'), (dval, 'eval')]
num_round = 100  # 迭代轮数

train_start_time = time.time()  # 记录训练开始时间
bst = xgb.train(params, dtrain, num_round, evals=watchlist, early_stopping_rounds=10)
train_end_time = time.time()  # 记录训练结束时间
print(f"Training time: {train_end_time - train_start_time:.2f} seconds")

# 3. 使用测试集进行预测并计算准确率
prediction_start_time = time.time()  # 记录预测开始时间
y_pred = bst.predict(dtest)
prediction_end_time = time.time()  # 记录预测结束时间
print(f"Prediction time on test set: {prediction_end_time - prediction_start_time:.2f} seconds")

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy * 100:.2f}%')

# 4. 计算其他评价指标
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision on test set: {precision:.2f}")
print(f"Recall on test set: {recall:.2f}")
print(f"F1-Score on test set: {f1:.2f}")
print("\nClassification Report on test set:\n", classification_report(y_test, y_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

# 5. 可视化：混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('xgboost混淆矩阵')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 6. 对测试数据集进行预测并写入Excel
prediction_ts_start_time = time.time()  # 记录预测开始时间
y_pred_encoded_ts = bst.predict(dtest_ts)
y_pred_labels_ts = label_encoder.inverse_transform(y_pred_encoded_ts.astype(int))
prediction_ts_end_time = time.time()  # 记录预测结束时间
print(f"Prediction time on external test set: {prediction_ts_end_time - prediction_ts_start_time:.2f} seconds")

# 读取并更新附件四
df_4 = pd.read_excel('附件四（Excel表）.xlsx')

# 将预测结果写入第二列（假设你想覆盖从第0到80行的第二列）
df_4.iloc[0:80, 1] = y_pred_labels_ts

# 保存结果到 Excel 文件
df_4.to_excel('./01_2/附件四（Excel表）_xgb.xlsx', index=False)

print("预测结果已保存至附件四（Excel表）_xgb.xlsx")
