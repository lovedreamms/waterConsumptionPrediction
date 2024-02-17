import numpy as np
import pandas as pd
names = ['A', 'B','C','D', 'E', 'F', 'G', 'H', 'I', 'J']
result = []
for name in names:
    df = pd.read_csv(f'Results/Result_{name}.csv')
    df['Date-time'] = pd.to_datetime(df['Date-time CET-CEST (DD/MM/YYYY HH:mm)'], format='%Y/%m/%d %H:%M:%S')
    df['Date-time'] = df['Date-time'].dt.strftime('%d/%m/%Y %H:%M')
    df.index = df['Date-time']
    result.append(df[f'DMA {name} (L/s)'].values)
result = pd.DataFrame(np.array(result).T, columns=[f'DMA {name} (L/s)' for name in names],index=df.index)
result.to_excel('Result.xlsx')