import numpy as np
import pre as pre
import random

"""
datagram的segment数量不一致，都统一为maxlen大小 空的填充0向量
不足：0的填充比较多
"""
def pad_data_x(data_xs, maxlen=100, PAD=0):  
    padded_data_xs = []
    for data_x in data_xs:
        # 一个datagram的
        if len(data_x) >= maxlen:
            padded_data_x = data_x[:maxlen]
        else:
            padded_data_x = data_x
            zero_len = maxlen-len(padded_data_x)
            zero = np.zeros([zero_len,8], int)
            padded_data_x = np.insert(padded_data_x, len(data_x), values=zero, axis=0)
            
        padded_data_xs.append(padded_data_x)
         
    return padded_data_xs

def read_dataset():
    num_classes = 3
    datagrams = pre.segment_slice()

    random.shuffle(datagrams) # 打乱数据集

    data_x = []
    data_y = []
    for datagram in datagrams:
        label = datagram[0]
        labels = [0] * num_classes
        labels[label-1] = 1 # 转化为one-hot
        data_y.append(labels)
        data_x.append(datagram[1])
    data_x = pad_data_x(data_x)

    # 划分训练集，验证集
    length = len(data_x)
    train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1 :]
    train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1 :]
    return train_x, train_y, dev_x, dev_y