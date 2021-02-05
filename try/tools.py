import torch
import torch.nn as nn

"""
将一个datagram的所有segment tensor化
features: (tensor)[[9,2,4,3,2...][...]...] 
labels: (tensor)[0,1,2,0,0,....]
"""
def make_tensor1(datagram):
    segment_num = len(datagram[1]) # 划分的segment数量
    labels = [datagram[0]]*segment_num  # 同一个datagram都是相同的segment
    #labels = [label]*segment_num # 同一个datagram都是相同的segment
    features = torch.LongTensor(datagram[1])
    labels = torch.LongTensor(labels)
    return features, labels

"""
features:[[1,2,....][....]...] datagram数量*150
labels: [0,1,2,.....] datagram数量
不足：理论上论文中不是截取150
"""
def make_tensor2(datas):
    labels = []
    features = []
    
    for data in datas:
        labels.append(data[0])
        features.append(data[1])
       
    features = pad_samples(features, maxlen=150)
    # 140*150 list
     
    features = torch.LongTensor(features)
    labels = torch.LongTensor(labels)
    return features, labels

"""
precission计算
"""
def precision(y_pred, y_true):
    # Calculates the precision
    true_positives = torch.sum(torch.clip(y_true * y_pred, 0, 1))
    predicted_positives = torch.sum(torch.clip(y_pred, 0, 1))
    precision = true_positives / predicted_positives
    return precision

"""
recall计算
"""
def recall(y_pred, y_true):
    # Calculates the recall
    true_positives = torch.sum(torch.clip(y_true * y_pred, 0, 1))
    possible_positives = torch.sum(torch.clip(y_true, 0, 1))
    recall = true_positives / possible_positives  
    return recall

"""
fbeta_score计算(论文中beta=1)
"""
def fbeta_score(y_pred, y_true, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    #y_true = torch.Tensor(one_hot_label(y_true))
    softmax = nn.Softmax(dim=1)
    y_pred = softmax(y_pred.view(-1,3))
    y_pred = y_pred.clamp(min=0.0001,max=1.0)
        
    y_true_ = torch.zeros(y_true.size(0), 3)
    y_true_.scatter_(1, y_true.view(-1, 1).long(), 1.)
    y_true = y_true_
     
    
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
 
    # If there are no true positives, fix the F score at 0 like sklearn.
    if torch.sum(torch.clip(y_true, 0, 1)) == 0:
        return 0
 
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r)
    return fbeta_score