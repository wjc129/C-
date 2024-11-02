import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis  # 导入计算偏度和峰度的函数

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_frequency_features(sequence_matrix, sampling_rate):
    num_sequences, sequence_length = sequence_matrix.shape
    frequency_features = []
    
    for i in range(num_sequences):
        sequence = sequence_matrix[i, :]
        
        # 1. 进行傅里叶变换
        fft_result = np.fft.fft(sequence)
        
        # 2. 计算频率分量的幅值
        fft_magnitude = np.abs(fft_result)
        
        # 3. 频率轴
        freqs = np.fft.fftfreq(sequence_length, d=1/sampling_rate)
        
        # 只取正频率部分
        positive_freqs = freqs[:sequence_length // 2]
        positive_magnitude = fft_magnitude[:sequence_length // 2]
        
        # 4. 提取频率相关特征：
        
        # (a) 主频率：幅值最大的频率
        dominant_frequency = positive_freqs[np.argmax(positive_magnitude)]
        
        # (b) 频谱能量：幅值平方的总和
        spectral_energy = np.sum(positive_magnitude ** 2)
        
        # (c) 谐波强度：除了主频率之外的其余频率分量的强度
        harmonic_strength = np.sum(positive_magnitude[positive_freqs != dominant_frequency] ** 2)
        
        # (d) 频谱熵：表示频谱随机性
        normalized_magnitude = positive_magnitude / np.sum(positive_magnitude)
        spectral_entropy = -np.sum(normalized_magnitude * np.log(normalized_magnitude + 1e-12))  # 避免log(0)
        
        # 汇总这些特征
        features = [dominant_frequency, spectral_energy, harmonic_strength, spectral_entropy]
        frequency_features.append(features)
        
    return np.array(frequency_features)

# 读取Excel文件的所有sheet
combined_df = pd.read_excel('sumtrain.xlsx')

# 提取波形采样点列的数据
wave_data = combined_df.iloc[:, 4:1028]
wave_data = wave_data.to_numpy()

# 提取波形的具体形状
wave_size = combined_df.iloc[:, 3] 
wave_size = wave_size.to_numpy()

# 统计特征矩阵
train_len = wave_data.shape[0]
staticfeactures = ['average', 'std', 'MaxVal', 'P2PVal', 'MinVal', 'skewness', 'kurtosis']  # 添加偏度和峰度
staticfeatures_df = pd.DataFrame(index=range(train_len), columns=staticfeactures)

# 计算统计特征
staticfeatures_df.iloc[:,0] = wave_data.mean(axis=1)   # 平均值
staticfeatures_df.iloc[:,1] = wave_data.std(axis=1)    # 标准差
staticfeatures_df.iloc[:,2] = wave_data.max(axis=1)    # 最大值
staticfeatures_df.iloc[:,4] = wave_data.min(axis=1)    # 最小值
staticfeatures_df.iloc[:,3] = wave_data.max(axis=1) - wave_data.min(axis=1)  # 峰峰值
staticfeatures_df.iloc[:,5] = skew(wave_data, axis=1)  # 偏度
staticfeatures_df.iloc[:,6] = kurtosis(wave_data, axis=1)  # 峰度

# 保存统计特征到Excel
staticfeatures_df.to_excel('./01/staticfeatures.xlsx', index=False)

# 频率特征矩阵
freqfeatures = ['dominant', 'spectral', 'harmonic', 'spectralentropy']

sampling_rate = 1000  # 采样率
# 提取频率相关特征
frequency_features = extract_frequency_features(wave_data, sampling_rate)
freqfeatures_df = pd.DataFrame(frequency_features, columns=freqfeatures)

# 保存频率特征到Excel
freqfeatures_df.to_excel('./01/freqfeatures.xlsx', index=False)

# 合并静态特征和频率特征
merged_df = pd.concat([staticfeatures_df, freqfeatures_df], axis=1)

# 插入励磁波形列
merged_df.insert(0, '励磁波形', wave_size)

# 保存最终特征到Excel
merged_df.to_excel('./01_2/classfyobject.xlsx', index=False)
