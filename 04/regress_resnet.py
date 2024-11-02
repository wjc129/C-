import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, x):
        return x + self.fc(x)  # 残差连接

# 定义 ResNet 模型
class ResNetRegression(nn.Module):
    def __init__(self, input_size, num_blocks=3):
        super(ResNetRegression, self).__init__()
        self.fc_in = nn.Linear(input_size, 256)  # 输入层
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_blocks)])  # 残差块
        self.fc_out = nn.Linear(256, 1)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        x = self.res_blocks(x)
        output = self.fc_out(x)
        return output

# 数据准备函数
def prepare_data(df):
    X = df.iloc[:, 1:].values  # 特征列
    y = df.iloc[:, 0].values   # 目标列

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 转换为 PyTorch 张量
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # 划分训练集（70%）和临时集（30%）
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # 从临时集中划分验证集（15%）和测试集（15%）
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
# 计算评价指标 RMSE, MAE, R²
def compute_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# 训练模型并保存最佳模型
def train_model_and_save(model, X_train, y_train, X_val, y_val, X_test, y_test, num_epochs=1000, learning_rate=0.0001, device='cpu', model_save_path='resnet_model.pth'):
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # 用于存储训练过程中每个 epoch 的指标
    train_rmse_list, val_rmse_list, test_rmse_list = [], [], []
    train_mae_list, val_mae_list, test_mae_list = [], [], []
    train_r2_list, val_r2_list, test_r2_list = [], [], []
    epochs_list = []

    # 追踪最佳验证集 RMSE 和最佳模型
    best_val_rmse = float('inf')
    best_model_state_dict = None
    epoch_best = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        # 每 100 个 epoch 计算并打印训练集、验证集和测试集的指标
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_rmse, train_mae, train_r2 = compute_metrics(y_train, outputs.squeeze())
                val_outputs = model(X_val).squeeze()
                val_rmse, val_mae, val_r2 = compute_metrics(y_val, val_outputs)
                test_outputs = model(X_test).squeeze()
                test_rmse, test_mae, test_r2 = compute_metrics(y_test, test_outputs)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
            print(f'Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}')
            print(f'Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}')

            # 记录每个 100 个 epoch 的指标
            train_rmse_list.append(train_rmse)
            val_rmse_list.append(val_rmse)
            test_rmse_list.append(test_rmse)
            train_mae_list.append(train_mae)
            val_mae_list.append(val_mae)
            test_mae_list.append(test_mae)
            train_r2_list.append(train_r2)
            val_r2_list.append(val_r2)
            test_r2_list.append(test_r2)
            epochs_list.append(epoch + 1)

            # 如果当前验证集 RMSE 比之前最佳 RMSE 更低，则保存当前模型
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state_dict = model.state_dict().copy()
                print(f'New best model found at epoch {epoch+1}, Val RMSE: {best_val_rmse:.4f}')
                epoch_best = epoch + 1

    print("Training completed.")

    # 保存最佳模型
    if best_model_state_dict is not None:
        torch.save({
            'model_state_dict': best_model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_save_path)
        print(f"Best model epoch {epoch_best} saved to {model_save_path} with Val RMSE: {best_val_rmse:.4f}")

    # 绘制 RMSE 图像
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_list, train_rmse_list, label="Train RMSE")
    plt.plot(epochs_list, val_rmse_list, label="Val RMSE")
    plt.plot(epochs_list, test_rmse_list, label="Test RMSE")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title("RMSE Over Time")
    plt.legend()
    plt.savefig('./04/rmse_over_time.png')
    plt.show()

    # 绘制 MAE 图像
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_list, train_mae_list, label="Train MAE")
    plt.plot(epochs_list, val_mae_list, label="Val MAE")
    plt.plot(epochs_list, test_mae_list, label="Test MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.title("MAE Over Time")
    plt.legend()
    plt.savefig('./04/mae_over_time.png')
    plt.show()

    # 绘制 R² 图像
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_list, train_r2_list, label="Train R²")
    plt.plot(epochs_list, val_r2_list, label="Val R²")
    plt.plot(epochs_list, test_r2_list, label="Test R²")
    plt.xlabel("Epochs")
    plt.ylabel("R²")
    plt.title("R² Over Time")
    plt.legend()
    plt.savefig('./04/r2_over_time.png')
    plt.show()

# 主程序
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 读取 Excel 文件
    df = pd.read_excel('./04/regressionobject.xlsx')  # 使用你的文件路径

    # 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)

    # 定义 ResNet 模型
    input_size = X_train.shape[1]
    model = ResNetRegression(input_size).to(device)

    # 训练模型并保存
    train_model_and_save(model, X_train, y_train, X_val, y_val, X_test, y_test, num_epochs=15000, learning_rate=0.005, device=device, model_save_path='./04/resnet_model.pth')
