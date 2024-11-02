import pandas as pd
import itertools
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot
import pingouin as pg
import scipy.stats as stats
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简体中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取数据
file_path = './03/Q3_data分析.xlsx'
excel_data = pd.ExcelFile(file_path)
df = excel_data.parse('Sheet1')

# 重命名列名，去除特殊字符和空格
df.rename(columns={'序号': 'id', '磁芯损耗，w/m3': '磁芯损耗'}, inplace=True)


# ANOVA方差分析
# 将温度、励磁波形、磁芯材料定义为分类变量
df['温度'] = df['温度'].astype('category')
df['励磁波形'] = df['励磁波形'].astype('category')
df['磁芯材料'] = df['磁芯材料'].astype('category')
df = df.drop(columns=['id'])  # 移除 id 列
# mapping = {
#     '温度': {'25度': 1, '50度': 2, '70度': 3, '90度': 4},
#     '励磁波形': {'正弦波': 1, '三角波': 2, '梯形波': 3},
#     '磁芯材料': {'材料1': 1, '材料2': 2, '材料3': 3, '材料4': 4}
# }
# df.replace(mapping, inplace=True)
# 描述性统计
for factor in ['温度', '励磁波形', '磁芯材料']:
    print(f"\n因素 '{factor}' 的描述性统计：")
    print(df.groupby(factor)['磁芯损耗'].describe())

# 单因素ANOVA
for factor in ['温度', '励磁波形', '磁芯材料']:
    model = ols(f'磁芯损耗 ~ C({factor})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"\n单因素ANOVA结果：因素 '{factor}'")
    print(anova_table)

# 两因素交互作用分析
factors = ['温度', '励磁波形', '磁芯材料']
combinations = list(itertools.combinations(factors, 2))

for combo in combinations:
    factor1, factor2 = combo
    formula = f'磁芯损耗 ~ C({factor1}) * C({factor2})'
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"\n两因素ANOVA结果：因素 '{factor1}' 和 '{factor2}'")
    print(anova_table)

factors = ['温度', '励磁波形', '磁芯材料']
for factor in factors:
    anova = pg.anova(dv='磁芯损耗', between=factor, data=df, detailed=True)
    print(f"\n效应量计算结果：因素 '{factor}'")
    print(anova[['Source', 'SS', 'DF', 'MS', 'F', 'p-unc', 'np2']])
    
combinations = list(itertools.combinations(factors, 2))

# 计算双因素的效应量
for combo in combinations:
    factor1, factor2 = combo
    anova = pg.anova(dv='磁芯损耗', between=[factor1, factor2], data=df, detailed=True)
    print(f"\n效应量计算结果：因素 '{factor1}' 和 '{factor2}'")
    print(anova[['Source', 'SS', 'DF', 'MS', 'F', 'p-unc', 'np2']])
# 可用的标记和颜色列表
all_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'X', '+', 'd']
all_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for combo in combinations:
    factor1, factor2 = combo
    

    x = df[factor1].astype(str)
    trace = df[factor2].astype(str)
    response = df['磁芯损耗']

    num_levels = trace.nunique()
    levels = trace.unique()

    # 确保标记和颜色数量足够
    if num_levels > len(all_markers):
        markers = all_markers * (num_levels // len(all_markers) + 1)
    else:
        markers = all_markers.copy()

    if num_levels > len(all_colors):
        colors = all_colors * (num_levels // len(all_colors) + 1)
    else:
        colors = all_colors.copy()

    # 截取所需数量的标记和颜色
    markers = markers[:num_levels]
    colors = colors[:num_levels]

    interaction_plot(x, trace, response,
                     colors=colors, markers=markers, ms=10)
    plt.xlabel(factor1)
    plt.ylabel('磁芯损耗均值')
    plt.title(f'交互作用图：{factor1} 和 {factor2}')
    plt.legend(title=factor2)
    plt.show()


# 构建包含所有因素和交互项的模型
formula = '磁芯损耗 ~ C(温度) * C(励磁波形) + C(温度) * C(磁芯材料) + C(励磁波形) * C(磁芯材料)'
model = ols(formula, data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\n完整模型的ANOVA结果：")
print(anova_table)

# 残差正态性检验
residuals = model.resid
stat, p = stats.shapiro(residuals)
print(f'\n残差正态性检验结果：Statistic={stat}, p-value={p}')
if p > 0.05:
    print('残差呈正态分布')
else:
    print('残差不呈正态分布，可能需要进行数据转换或使用非参数检验。')
    

# 1. 计算所有因素水平组合下的磁芯损耗均值
grouped_means = df.groupby(['温度', '励磁波形', '磁芯材料'])['磁芯损耗'].mean().reset_index()

# 2. 按磁芯损耗升序排序，找到磁芯损耗最小的因素组合
sorted_means = grouped_means.sort_values(by='磁芯损耗', ascending=True)
print("磁芯损耗最小的因素组合（根据实际数据均值）：")
print(sorted_means.head())

# 3. 使用模型预测磁芯损耗最小的因素组合

# 获取每个因素的水平
temperatures = df['温度'].cat.categories
waveforms = df['励磁波形'].cat.categories
materials = df['磁芯材料'].cat.categories

factor_combinations = list(itertools.product(temperatures, waveforms, materials))

# 创建包含所有组合的DataFrame
df_combinations = pd.DataFrame(factor_combinations, columns=['温度', '励磁波形', '磁芯材料'])

# 使用模型进行预测
# 注意：模型需要完整的数据，包括处理过的分类变量
# 我们需要将df_combinations中的分类变量转换为与模型一致的格式

# 首先，将分类变量转换为category类型，确保类别一致
df_combinations['温度'] = df_combinations['温度'].astype('category')
df_combinations['励磁波形'] = df_combinations['励磁波形'].astype('category')
df_combinations['磁芯材料'] = df_combinations['磁芯材料'].astype('category')

# 确保类别的顺序与原数据一致
# 确保类别的顺序与原数据一致
df_combinations['温度'] = df_combinations['温度'].cat.set_categories(df['温度'].cat.categories)
df_combinations['励磁波形'] = df_combinations['励磁波形'].cat.set_categories(df['励磁波形'].cat.categories)
df_combinations['磁芯材料'] = df_combinations['磁芯材料'].cat.set_categories(df['磁芯材料'].cat.categories)


# 使用模型预测
predicted = model.predict(df_combinations)

# 将预测结果添加到DataFrame
df_combinations['预测磁芯损耗'] = predicted

# 按预测磁芯损耗升序排序
df_combinations_sorted = df_combinations.sort_values(by='预测磁芯损耗')
print("\n根据模型预测，磁芯损耗最小的因素组合：")
print(df_combinations_sorted.head())

# 为了可视化，我们可以绘制热力图，展示在不同温度和励磁波形下，不同磁芯材料的磁芯损耗
# 需要将数据整理成透视表格式

# 使用实际数据的均值
pivot_table = grouped_means.pivot_table(values='磁芯损耗', index='磁芯材料', columns=['温度', '励磁波形'], observed=False)


# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('不同因素组合下的磁芯损耗均值（实际数据）')
plt.xlabel('温度和励磁波形组合')
plt.ylabel('磁芯材料')
plt.savefig('./03/不同因素组合下的磁芯损耗均值（实际数据）.png')
plt.show()

# 使用模型预测的结果
pivot_table_pred = df_combinations.pivot_table(values='预测磁芯损耗', index='磁芯材料', columns=['温度', '励磁波形'],observed=False)

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table_pred, annot=True, fmt=".1f", cmap="YlOrRd")
plt.title('不同因素组合下的磁芯损耗预测值（模型预测）')
plt.xlabel('温度和励磁波形组合')
plt.ylabel('磁芯材料')
plt.savefig('./03/不同因素组合下的磁芯损耗预测值（模型预测）.png')
plt.show()

