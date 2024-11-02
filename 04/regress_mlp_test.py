import torch
import pandas as pd
from regress_mlp import MLPRegression  # 从训练文件导入模型
import numpy as np
from sklearn.preprocessing import StandardScaler

# 加载模型并进行预测
def load_model_and_predict(model, X_data, model_load_path='mlp_model.pth', device='cpu'):
    # 加载模型
    model.load_state_dict(torch.load(model_load_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 将数据迁移到设备上
    X_data = X_data.to(device)
    
    # 将数据移动到 CPU 并转换为 NumPy 数组进行标准化
    X_data_np = X_data.cpu().numpy()
    scaler = StandardScaler()
    X_data_scaled = scaler.fit_transform(X_data_np)

    # 转换为张量
    X_data_scaled = torch.tensor(X_data_scaled, dtype=torch.float32).to(device)

    # 进行预测
    with torch.no_grad():
        predictions = model(X_data_scaled).squeeze().cpu().numpy()
    return predictions
# 主程序
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_excel('./04/regressionobject_test.xlsx')  # 使用新的特征数据文件路径

    X_data = torch.tensor(df.values, dtype=torch.float32)
    
    # 定义 MLP 模型
    input_size = X_data.shape[1]  # 特征的维度
    model = MLPRegression(input_size).to(device)

    # 加载模型并进行预测
    predictions = load_model_and_predict(model, X_data, model_load_path='./04/mlp.pth', device=device)
    
    # 将预测结果添加到 DataFrame 中作为新的一列
    df['Predicted Values'] = predictions

    # 输出更新后的 DataFrame
    print(df.head())  # 显示前几行结果
    
    # 如果需要将结果保存到一个新的 Excel 文件中
    df.to_excel('./04/predicted_results.xlsx', index=False)
    print("Predictions have been saved to './04/predicted_results.xlsx'.")
