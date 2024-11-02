import pandas as pd

sheet1 = pd.read_excel('附件一（训练集）.xlsx', sheet_name='材料1')
sheet2 = pd.read_excel('附件一（训练集）.xlsx', sheet_name='材料2')
sheet3 = pd.read_excel('附件一（训练集）.xlsx', sheet_name='材料3')
sheet4 = pd.read_excel('附件一（训练集）.xlsx', sheet_name='材料4')

sheet1['材料'] = 1
sheet2['材料'] = 2
sheet3['材料'] = 3
sheet4['材料'] = 4

sheets = [sheet1, sheet2, sheet3, sheet4]
combined_df = pd.concat(sheets,axis=0,ignore_index=True)
combined_df.to_excel('./04/traindata.xlsx', index=False)