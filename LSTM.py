import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.5
    else:
        return lr
class TestModelCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TestModelCallback, self).__init__()
        self.res = [0.6]

    def on_epoch_end(self, epoch, logs=None):
        # 在这里执行测试
        res = test()
        if np.array(self.res).max() < res :
            model_filename = 'model_'+f'{name}_' + str(int(res*10000)) + '.h5'
            # 保存模型
            model.save(model_filename)
        self.res.append(res)
def test():
    test_predict = model.predict(X_test)
    num_features = scaled_data.shape[1]
    test_predict_extended = np.zeros((len(test_predict), num_features))
    test_predict_extended[:, -1] = test_predict.ravel() 
    y_test_extended = np.zeros((len(y_test), num_features))
    y_test_extended[:, -1] = y_test.ravel()  # 确保y_test是正确的形状
    test_predict_inversed = scaler.inverse_transform(test_predict_extended)[:, -1]
    y_test_inversed = scaler.inverse_transform(y_test_extended)[:, -1]
    nse_value = nash_sutcliffe_efficiency(y_test_inversed, test_predict_inversed)
    print(f"去归一化后的NSE值: {(1-nse_value/1000)}")
    return 1-nse_value/1000
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

def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), 
                   return_sequences=True, 
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), 
                   return_sequences=False))
    model.add(Dense(hp.Int('dense_units', min_value=16, max_value=256, step=16)))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')
    
    return model

names = ['A', 'B','C','D', 'E', 'F', 'G', 'H', 'I', 'J']

for name in names:

# 加载数据
    df = pd.read_csv(f'data/Data{name}.csv',index_col=0)
    df.dropna(inplace=True)
    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    print(scaled_data)
    with open(f'data/Scaler{name}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    time_step = 8
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    # 划分训练集和测试集
    split_frac = 0.8  # 80%作为训练集
    split = int(split_frac * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),kernel_regularizer=l1(0.01),))
    model.add(LSTM(200, return_sequences=True,kernel_regularizer=l1(0.01),))
    model.add(LSTM(200, return_sequences=True,kernel_regularizer=l1(0.01),))
    model.add(LSTM(200, return_sequences=False,kernel_regularizer=l1(0.01),))
    model.add(Dense(256,kernel_regularizer=l1(0.01)))
    model.add(Dense(128, kernel_regularizer=l1(0.01)))  
    model.add(Dense(64, kernel_regularizer=l1(0.01))) 
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss=nash_sutcliffe_efficiency)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=100, verbose=1,callbacks=[TestModelCallback(), LearningRateScheduler(scheduler)])

    # 预测
    # train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # 绘制结果
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Original Test Data')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), test_predict, label='Predicted Test Data')
    plt.legend()
    plt.show()
    model_filename = 'model_' + name+'_' + 'last' + '.h5'
    model.save(model_filename)



