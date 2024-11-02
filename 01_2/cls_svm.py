import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
# 记录开始时间
start_time = time.time()

# 1. 数据预处理
preprocess_start_time = time.time()
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
preprocess_end_time = time.time()
print(f"Data preprocessing time: {preprocess_end_time - preprocess_start_time:.2f} seconds")

# 2. 使用SVM进行训练
train_start_time = time.time()
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)  # 'rbf' 核函数，C和gamma可以调优
svm_model.fit(X_train, y_train)
train_end_time = time.time()
print(f"Training time: {train_end_time - train_start_time:.2f} seconds")

# 3. 使用验证集进行预测并计算准确率
val_pred_start_time = time.time()
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_pred_end_time = time.time()
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
print(f"Validation prediction time: {val_pred_end_time - val_pred_start_time:.2f} seconds")

# 4. 使用测试集进行预测并计算准确率
test_pred_start_time = time.time()
y_test_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
test_pred_end_time = time.time()
print(f'Accuracy on test set: {accuracy * 100:.2f}%')
print(f"Test prediction time: {test_pred_end_time - test_pred_start_time:.2f} seconds")

# 5. 计算其他评价指标
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Precision on test set: {precision:.2f}")
print(f"Recall on test set: {recall:.2f}")
print(f"F1-Score on test set: {f1:.2f}")
print("\nClassification Report on test set:\n", classification_report(y_test, y_test_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

# 6. 可视化：混淆矩阵
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('SVM混淆矩阵')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 7. 对测试数据集进行预测并写入Excel
ts_pred_start_time = time.time()
y_pred_ts = svm_model.predict(scaled_features_ts)
y_pred_labels_ts = label_encoder.inverse_transform(y_pred_ts)
ts_pred_end_time = time.time()
print(f"Prediction time on external test set: {ts_pred_end_time - ts_pred_start_time:.2f} seconds")

# 读取并更新附件四
df_4 = pd.read_excel('附件四（Excel表）.xlsx')

# 将预测结果写入第二列（假设你想覆盖从第0到80行的第二列）
df_4.iloc[0:80, 1] = y_pred_labels_ts

# 保存结果到 Excel 文件
df_4.to_excel('./01_2/附件四（Excel表）_svm.xlsx', index=False)

# 总耗时
end_time = time.time()
print("预测结果已保存至附件四（Excel表）_svm.xlsx")
print(f"Total execution time: {end_time - start_time:.2f} seconds")
