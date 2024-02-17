import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 假设您已经按照之前的代码读取和处理了数据
df1 = pd.read_excel('data\WaterUsageData.xlsx',index_col=0)
df2 = pd.read_excel('data\WeatherData.xlsx',index_col=0)
name = 'F'
df = pd.merge(df1,df2,left_index=True,right_index=True,how='outer')

# 确保索引是日期时间格式，如果不是，需要转换
df.index = pd.to_datetime(df.index)

# 按天聚合数据，这里以'DMA C (L/s)'为例，计算每天的平均值
daily_data = df.resample('C').mean()

# 绘制'DMA C (L/s)'的日变化情况
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(daily_data.index, daily_data[f'DMA {name} (L/s)'], marker='o', linestyle='-')
plt.title(f'Daily Variation of DMA {name} (L/s)')  # 设置图形标题
plt.xlabel('Date')  # 设置x轴标签
plt.ylabel(f'DMA {name} (L/s)')  # 设置y轴标签
plt.xticks(rotation=45)  # 旋转x轴标签，以便更好地显示日期
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()
