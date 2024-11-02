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
        fft_result = np.fft.fft(sequence)
        fft_magnitude = np.abs(fft_result)
        freqs = np.fft.fftfreq(sequence_length, d=1/sampling_rate)
        positive_freqs = freqs[:sequence_length // 2]
        positive_magnitude = fft_magnitude[:sequence_length // 2]
        dominant_frequency = positive_freqs[np.argmax(positive_magnitude)]
        spectral_energy = np.sum(positive_magnitude ** 2)
        harmonic_strength = np.sum(positive_magnitude[positive_freqs != dominant_frequency] ** 2)
        normalized_magnitude = positive_magnitude / np.sum(positive_magnitude)
        spectral_entropy = -np.sum(normalized_magnitude * np.log(normalized_magnitude + 1e-12))  
        features = [dominant_frequency, spectral_energy, harmonic_strength, spectral_entropy]
        frequency_features.append(features)
    return np.array(frequency_features)

combined_df = pd.read_excel('附件二（测试集）.xlsx')
wave_data = combined_df.iloc[:, 4:1028]
wave_data = wave_data.to_numpy()

wave_size = combined_df.iloc[:, 3].to_numpy()

train_len = wave_data.shape[0]
staticfeactures = ['average', 'std', 'MaxVal', 'P2PVal', 'MinVal', 'skewness', 'kurtosis']  # 添加偏度和峰度

staticfeatures_df = pd.DataFrame(index=range(train_len), columns=staticfeactures)
staticfeatures_df['average'] = wave_data.mean(axis=1)   # 平均值
staticfeatures_df['std'] = wave_data.std(axis=1)        # 标准差
staticfeatures_df['MaxVal'] = wave_data.max(axis=1)     # 最大值
staticfeatures_df['MinVal'] = wave_data.min(axis=1)     # 最小值
staticfeatures_df['P2PVal'] = wave_data.max(axis=1) - wave_data.min(axis=1)  # 峰峰值
staticfeatures_df['skewness'] = skew(wave_data, axis=1)  # 偏度
staticfeatures_df['kurtosis'] = kurtosis(wave_data, axis=1)  # 峰度
staticfeatures_df.to_excel('./01/staticfeatures_test.xlsx', index=False)
freqfeatures = ['dominant', 'spectral', 'harmonic', 'spectralentropy']
sampling_rate = 1000  # 采样率
frequency_features = extract_frequency_features(wave_data, sampling_rate)
freqfeatures_df = pd.DataFrame(frequency_features, columns=freqfeatures)
freqfeatures_df.to_excel('./01/freqfeatures_test.xlsx', index=False)
merged_df = pd.concat([staticfeatures_df, freqfeatures_df], axis=1)
merged_df.to_excel('./01_2/classfyobject_test.xlsx', index=False)
