import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_excel('data\WaterUsageData.xlsx',index_col=0)
df2 = pd.read_excel('data\WeatherData.xlsx',index_col=0)
name = 'F'
df = pd.merge(df1,df2,left_index=True,right_index=True,how='outer')
# 确保索引是日期时间格式，如果不是，需要转换
df.index = pd.to_datetime(df.index)
df.dropna(inplace=True)
first_month_data = df[df.index.month == 1].index.year[0]
first_month_data = df[(df.index.month == 1) & (df.index.year == first_month_data)& (df.index.day ==5)]
print(first_month_data.shape)
plt.figure(figsize=(15, 7))  # 设置图形大小
plt.plot(first_month_data.index, first_month_data[f'DMA {name} (L/s)'], marker='o', linestyle='-', markersize=3)
plt.title(f'Hourly Variation of DMA {name} (L/s) in the First Month')  # 设置图形标题
plt.xlabel('Date and Hour')  # 设置x轴标签
plt.ylabel(f'DMA {name} (L/s)')  # 设置y轴标签
plt.xticks(rotation=45)  # 旋转x轴标签，以便更好地显示日期和时间
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()
