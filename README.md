## 使用

先运行 result.py 生成结果数据，保存在 Result 文件夹下
然后运行 Resulttoexcel.py 即可生成结果所需文件 Result.py

## 训练

LSTM.py 是模型的训练文件

## data 文件夹

### Data\_

Data\_ 是数据文件夹，里面包含了数据文件和数据说明文件
由 dataProcess.py 生成 有不同的格式，用于 LSTM.py 和 result.py 的数据输入

### Scaler

由 LSTM.py 保存的数据 minmaxScaler 标准化器，在 result.py 做数据反标准化

## models 文件夹

保存各个地区的模型，由 LSTM.py 生成 正确率有高有底
在我的 2023-2-28 的 dailynote 由正确率记录

## my_dir 文件夹

不重要，是使用网格搜索寻找最优参数的生成文件
