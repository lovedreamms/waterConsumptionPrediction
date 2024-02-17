import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import keras.utils
# from LSTM import nash_sutcliffe_efficiency

def nash_sutcliffe_efficiency(y_obs, y_pred):
    """
    使用 TensorFlow 操作计算纳什效率系数(NSE)

    参数:
    y_obs: 实际观测值的张量
    y_pred: 模型预测值的张量

    返回:
    NSE值
    """
    # 确保y_obs和y_pred为Tensor
    y_obs = tf.convert_to_tensor(y_obs, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    
    # 计算观测值的平均值
    mean_obs = tf.reduce_mean(y_obs)
    
    # 计算分子（预测误差的平方和）
    numerator = tf.reduce_sum(tf.square(y_obs - y_pred))
    
    # 计算分母（观测值与其平均值差的平方和）
    denominator = tf.reduce_sum(tf.square(y_obs - mean_obs))
    # print(numerator, denominator)
    # 计算NSE
    nse = 1000*(numerator / denominator) 
    
    return nse
# 创建数据集的函数

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), :]  # 假设最后一列是目标变量
        X.append(a)
        Y.append(data[i + time_step, -1])
    return np.array(X), np.array(Y)

# keras.utils.get_custom_objects()['nash_sutcliffe_efficiency'] = nash_sutcliffe_efficiency
names = ['A', 'B','C','D', 'E', 'F', 'G', 'H', 'I', 'J']
for name in names:
    # df1 = pd.read_excel('data\WaterUsageData.xlsx',index_col=0)
    df = pd.read_csv(f'data\Data{name}.csv',index_col=0)
    # df = pd.merge(df1,df2,left_index=True,right_index=True,how='outer')
    df = df.loc['2023-06-30 16:00:00':'2023-07-31 23:00:00',:]
    # df.index = range(len(df))
    with keras.utils.custom_object_scope({'nash_sutcliffe_efficiency': nash_sutcliffe_efficiency}):
        model = keras.models.load_model(f'models/model_{name}.h5')  
    # print(df.head(10))
    for time in range(8,752):
        x = df.iloc[time-8:time,:7]
        with open(f'data/Scaler{name}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        # 在新文件中使用加载的标准化对象
        x = scaler.transform(x)
        # 加载模型时注册自定义损失函数
        x = np.array(x)
        x = x.reshape(1,x.shape[0], x.shape[1])
        test_predict = model.predict(x)  
        result = scaler.inverse_transform([[0,0,0,0,0,0,test_predict]])[0][-1]
        # 填入数据 进行继续预测
        df.iloc[time,6] = result
    df = df.loc['2023-07-01 00:00:00':'2023-07-31 23:00:00',:]
    df.to_csv(f'Results/Result_{name}.csv')
    print('保存一个文件')
        
