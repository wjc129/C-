import pandas as pd
import numpy as np
# 主频不再纳入特征
# 读取traindata.xlsx

combined_df = pd.read_excel('./04/testdata.xlsx')
# 提取波形采样点列的数据
wave_data = combined_df.iloc[:, 5:1029]
print(wave_data.shape)
wave_data = wave_data.to_numpy()
print(wave_data.shape[1])
print(wave_data[0].shape)
print(wave_data[0])
# 提取波形的具体形状
wave_size = combined_df.iloc[:, 4] 
wave_size = wave_size.to_numpy()

# 统计特征矩阵
train_len = wave_data.shape[0]
print(train_len)
staticfeactures = ['MaxVal']
staticfeatures_df = pd.DataFrame(index=range(train_len), columns=staticfeactures)
staticfeatures_df.iloc[:,0] = wave_data.max(axis=1)

merged_df = staticfeatures_df

#tempre 代表温度
temre = combined_df.iloc[:, 1].to_numpy()
#meter 代表材料
meter = combined_df.iloc[:, 3].to_numpy()
#freq 代表频率
freq = combined_df.iloc[:, 2].to_numpy()

merged_df['材料'] = meter
merged_df['温度'] = temre
merged_df['频率'] = freq
merged_df['励磁波形'] = wave_size
merged_df.to_excel('./04/regressionobject_test.xlsx', index=False)


