import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
names = ['A', 'B','C','D', 'E', 'F', 'G', 'H', 'I', 'J']

for name in names:
    df1 = pd.read_excel('data\WaterUsageData.xlsx',index_col=0)
    df2 = pd.read_excel('data\WeatherData.xlsx',index_col=0)
    df = pd.merge(df1,df2,left_index=True,right_index=True,how='outer')
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    print(df.columns)
    data = df[['hour_sin','hour_cos','Rainfall depth (mm)','Air temperature (°C)', 'Air humidity (%)', 'Windspeed (km/h)',f'DMA {name} (L/s)']]
    # data = df[['hour_sin','hour_cos','Rainfall depth (mm)','Air temperature (°C)', 'Air humidity (%)', 'Windspeed (km/h)',f'DMA {name} (L/s)']]
    # data.dropna(inplace=True)
    data.to_csv(f'data\Data{name}.csv',index=True)
    # df.to_csv('data\Data.csv')

