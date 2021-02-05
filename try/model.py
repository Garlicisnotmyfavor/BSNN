import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data  
import matplotlib.pyplot as plt
import torch.nn.functional as F
import tools as tl
import numpy as np

"""
使用LSTM unit的attention nn
"""
class BiLSTM_Attention(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, n_layers, class_dim):

        super(BiLSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, class_dim) # 3指有三个分类label
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(256, hidden_dim)

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


    def attention_net(self, x):       #x:[batch, seq_len, hidden_dim*2]

        u = torch.tanh(torch.matmul(x, self.w_omega))         #[batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)                   #[batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)

        scored_x = x * att_score                              #[batch, seq_len, hidden_dim*2]

        context = torch.sum(scored_x, dim=1)                  #[batch, hidden_dim*2]
        return context


    def forward(self, x):
        embedding = self.dropout(self.embedding(x))       #[seq_len, batch, embedding_dim]

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        #output = output.permute(1, 0, 2)  #[batch, seq_len, hidden_dim*2]
        
        attn_outputs = self.attention_net(output)
        logit = self.fc(attn_outputs)
        return logit

"""
FocalLoss
"""
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha)

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)
        
        target_ = torch.zeros(target.size(0),self.class_num)
        target_.scatter_(1, target.view(-1, 1).long(), 1.)
        
        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

"""
train数据函数
"""
def train(rnn, iterator, optimizer, criteon):

    avg_loss = []
    avg_acc = []
    rnn.train()        #表示进入训练模式

    for i, batch in enumerate(iterator):
        pred = rnn(batch[0])             #[batch, 1] -> [batch]
        
        loss = criteon(pred, batch[1])
        acc = tl.fbeta_score(pred, batch[1]).item()   #计算每个batch的准确率

        avg_loss.append(loss.item())
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc

"""
评估数据函数
"""
def evaluate(rnn, iterator, criteon):

    avg_loss = []
    avg_acc = []
    rnn.eval()         #表示进入测试模式

    with torch.no_grad():
        for batch in iterator:

            pred = rnn(batch[0])       #[batch, 1] -> [batch]
            #print(pred)
            #print(batch[1])

            loss = criteon(pred, batch[1])
            acc = tl.fbeta_score(pred, batch[1]).item()

            avg_loss.append(loss.item())
            avg_acc.append(acc)

    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc

"""
    :param logits:  [batch_size, n_class]
    :param labels: [batch_size]  not one-hot !!!
    :return: -alpha*(1-y)^r * log(y)
    """
def focal_loss(labels, logits, alpha=[[1], [1], [1]], epsilon = 1.e-7, gamma=2.0):
    label_ = []
    print(labels)
    for label in labels:
        if(label[0] == 1):
            label_.append(0)
        if(label[1] == 1):
            label_.append(1)
        if(label[2] == 1):
            label_.append(2)
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, [-1, logits.shape[2]])
        labels = tf.reshape(label_, [-1])
        # (Class ,1)
        alpha = tf.constant(alpha, dtype=tf.float32)
        labels = tf.cast(label_, dtype=tf.int32)
        logits = tf.cast(logits, tf.float32)
        # (N,Class) > N*Class
        softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
        # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
        labels_shift = tf.range(0, logits.shape[0]) * logits.shape[1] + label_
        #labels_shift = tf.range(0, batch_size*32) * logits.shape[1] + labels
        # (N*Class,) > (N,)
        prob = tf.gather(softmax, labels_shift)
        # 预防预测概率值为0的情况  ; (N,)
        prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
        # (Class ,1) > (N,)
        alpha_choice = tf.gather(alpha, label_)
        # (N,) > (N,)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        weight = tf.multiply(alpha_choice, weight)
        # (N,) > 1
        loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
        return loss