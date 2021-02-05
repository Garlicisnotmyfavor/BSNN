# BSNN复现

### 论文理解

本质其实是一个文本分类问题，解决的问题是对网络协议分类，但不同于以往的协议分类问题，这篇论文不使用报文头等特征信息，而是纯粹地从payload特征提取角度对网络协议分类。因此可以看作一个文本分类问题，输入一个packet的payload部分，输出报文协议类型。其优点在于：

- 可以学习新的协议，而不被限制于已有协议
- 不需要繁复的对报文头特征信息的挖掘（并且这些信息可能是不靠谱的，可以修改的话）

<img src="http://garlicisnotmyfavor.xyz/2021/02/05/BSNN/1.png" width = "450" height = "250" align=center />

模型结构如上所示，是一个Hierarchical attention network。

先解释一下模型中的各个字段的意思。Datagram指的是一个packet中的payload部分，如下图wireshark抓包后蓝色的byte部分。进一步对Datagram做划分可以得到很多个长度均等的Segment，这里如果设定每个segment长度都为5的话，一个segment就可以表达为下图中红色方框部分。可以看到一个segment[3b, de, 01, 00, 00]里有多个十六进制字符。

这里为了更好的理解，可以做一个类比：

- 一篇文章 ==>  datagram ==> [3b, de, 01, 00, 00, 01, 00, 00......]
- 文章中的句子 ==> segment ==> [3b, de, 01, 00, 00]
- 句子中的单词 ==> 3b

<img src="http://garlicisnotmyfavor.xyz/2021/02/05/BSNN/2.png" width = "450" height = "250" align=center />

现在要对这样一个结构的datagram做训练，最后需要学到它所属的协议类型。模型的直觉就是我们首先去关注一个句子（segment）中单词（character）的表达，使用rnn(LSTM/GRU)训练并且每个单词对于这个句子的重要性是不同的（attention机制），通过这样一个过程得到这个句子（segment）的表达。之后再重复类似的过程，在一篇文章（datagram）中，不同句子的重要性不同，关注于“焦点”，通过一个attention encoder得到这篇文章（datagram）的表达，使用softmax等最后得到这个文章（datagram）的类别。

整个模型有两层attention encoder，使用的RNN Unit是LSTM/GRU（我后面实现的GRU，因为对我的电脑友好一些🐶），且是bidirectional的（双向），因为这里上下文都是有意义的，需要全局的。实现细节上使用Focal Loss，主要是为了处理样本数据不均衡的问题（好巧不巧我做数据的时候正好每个类别都是均分的，似乎对我这个没有多大用处，我的错），评估指标使用$F_1$。

### 代码复现

代码目录中data文件夹内为数据，try文件夹中为第一次尝试的pytorch代码（nn结构有问题，当时不太搞明白多层attention怎么写，但也是个实践过程先保存下来），run_BSNN保存的运行数据，pre.py做数据预处理，DataUtil.py做数据padding及划分，BSNN.py放模型，train.py为运行入口。

tensorflow版本为1.14（主要想使用一个老版本的库，降到这个版本，warning还是有的😅），pre使用的pytorch（但是相关数据预处理已运行出来保存好了，因此不需要再使用pytorch运行pre.py）

#### 数据预处理

论文中数据集是自己收集了，并且在网上找了一下现存数据集，比较难得到条件适合的，因此自己流量抓包收集了300笔数据（多了我电脑伤不起欸😭）。数据特征如下：

- 抓的是DNS，OICQ，QUIC三个协议，每个抓取了100笔，其中后两个是google使用的快传协议和QQ使用的协议
- 一些报文内容较短，一些很长，这个特点使得做数据padding时较难选择一个合适的统一长度（我这里最后选择的100作为固定长）
- 因为是连续地抓取，前后报文相似度很多，因此每类协议多样性不足

预处理流程如下图所示：

<img src="http://garlicisnotmyfavor.xyz/2021/02/05/BSNN/3.png" width = "450" height = "250" align=center />

pre.py中数据处理代码

```python
# -*- coding: utf-8 -*-
import json
import csv
import pandas as pd
import numpy as np

# 一个segment的长度，论文中推荐的是L=8
SEGMENT_LEN = 8

packets = []

# 下面三个都是对报文数据做处理，提取出需要的payload部分，存储下来
with open('data/QUIC.json', 'r', encoding='UTF-8') as reader:
    content = json.load(reader)
    #len(content) 先弄小一点的数据集
    for i in range(0, 100):
        packet = ['0', content[i]["_source"]["layers"]["udp"]["udp.payload"]]
        packets.append(packet)
    reader.close()

# QQ的协议
with open('data/OICQ.json', 'r', encoding='UTF-8') as reader:
    content = json.load(reader)
    #len(content) 先弄小一点的数据集
    for i in range(0, 100):
        packet = ['1', content[i]["_source"]["layers"]["udp"]["udp.payload"]]
        packets.append(packet)
    reader.close()

with open('data/DNS.json', 'r', encoding='UTF-8') as reader:
    content = json.load(reader)
    #len(content) 先弄小一点的数据集
    for i in range(0, 100):
        packet = ['2', content[i]["_source"]["layers"]["udp"]["udp.payload"]]
        packets.append(packet)
    reader.close()

# train数据写入csv ==> payload.csv
csv_fp = open("data/train.csv", "w", encoding='utf-8', newline='')
writer = csv.writer(csv_fp)
writer.writerow(['label', 'payload'])
writer.writerows(packets)
csv_fp.close()

"""
将payload中字符转化为0-255数字
将每个datagram的payload按照长度N=SEGMENT_LEN划分
return: [[label, [[0,1,2,3,...][]....]]]
"""
def segment_slice():
    data = pd.read_csv('data/train.csv', encoding='UTF-8')
    labels = data['label']
    payloads = data['payload'].apply(lambda x :x.split(':'))
    Datagrams = [] # [[label, [0,255,...]],...]
    for i in range(0, len(data)):
        Datagram = [] # [label, [0,255,...]]
        for j in range(0, len(payloads[i])):
            Datagram.append(int(payloads[i][j], 16))
        Datagrams.append([labels[i], Datagram])

    # 切片
    for i in range(0, len(Datagrams)):
        last_num = len(Datagrams[i][1]) % SEGMENT_LEN
        Datagrams[i][1] = Datagrams[i][1] + [0]*(SEGMENT_LEN-last_num)
        temp = np.array(Datagrams[i][1])
        temp = temp.reshape(-1, SEGMENT_LEN)
        Datagrams[i][1] = temp

    return Datagrams
```

DataUtil.py中进一步处理

```python
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
```

#### 模型构建

模型主要分为四个部分：

- word encoder （BiGRU layer）
- word attention （Attention layer）
- sentence encoder （BiGRU layer）
- sentence attention （Attention layer）

##### GRU原理

<img src="http://garlicisnotmyfavor.xyz/2021/02/05/BSNN/4.png" width = "450" height = "250" align=center />

GRU是RNN的一个变种，使用门机制来记录当前序列的状态。在GRU中有两种类型的门（gate）: reset gate和update gate。这两个门一起控制来决定当前状态有多少信息要更新。

reset gate是用于决定多少过去的信息被用于生成候选状态，如果Rt为0，表明忘记之前的所有状态：
$$
r_t = \sigma(W_rx_r+U_rh_{y-1}+b_r)
$$
根据reset gate的分数可以计算出候选状态：
$$
\hat{h_t} = tanh(W_hx_t+r_t\oplus(U_hh_{t-1}+b_h))
$$
update gate是用来决定由多少过去的信息被保留，以及多少新信息被加进来：
$$
z_t = \sigma(W_zx_r+U_zh_{t-1}+b_z)
$$
最后，隐藏层状态的计算公式，有update gate、候选状态和之前的状态共同决定：
$$
h_t = (1-z_t)\oplus h_{t-1}+z_t \oplus \hat{h_t}
$$

##### Attention原理

<img src="http://garlicisnotmyfavor.xyz/2021/02/05/BSNN/5.png" width = "450" height = "250" align=center />

##### 1、word encoder layer

首先，将每个segment中的character做embedding转换成向量（这里使用one-hot），然后输入到双向GRU网络中，结合上下文的信息，获得该character对应的隐藏状态输出$h_{it}=[\overrightarrow{h_{it}}, \overleftarrow{h_{it}}]$
$$
x_{it} = W_cw_{it}, t \in [i,T]
$$

$$
\overrightarrow{h_{it}} = \overrightarrow{GRU}(x_{it})
$$

$$
\overleftarrow{h_{it}} = \overleftarrow{GRU}(x_{it})
$$

##### 2、word attention layer

attention机制的目的就是要把一个segment中，对segment表达最重要的character找出来，赋予一个更大的比重。

首先将word encoder那一步的输出得到的$h_{it}$输入到一个单层的感知机中得到结果$u_{it}$作为其隐含表示
$$
u_{it} = tanh(W_wh_{it}+b_w)
$$
接着为了衡量character的重要性，定义了一个随机初始化的character层面上下文向量$u_w$，计算其与segment中每个charater的相似度，然后经过一个`softmax`操作获得了一个归一化的attention权重矩阵$\alpha_{it}$，代表segement i中第t个charater的权重：

$$
\alpha_{it} = \frac {exp(u_{it}^Tu_w)}{\sum_t(exp(u_{it}^Tu_w))}
$$
于是，segment的向量$s_i$就可以看做是segment中character的向量的加权求和。这里的character层面上下文向量是$u_w$随机初始化并且可以在训练的过程中学习得到的。
$$
s_i = \sum_{t} \alpha_{it}h_{it}
$$

##### 3、sentence encoder

通过上述步骤我们得到了每个segment的向量表示，然后可以用相似的方法得到datagram向量$h_{i}=[\overrightarrow{h_{i}}, \overleftarrow{h_{i}}]$：
$$
\overleftarrow{h_{i}} = \overleftarrow{GRU}(s_{i}), i \in [i,L]
$$

$$
\overrightarrow{h_{i}} = \overrightarrow{GRU}(s_{i}), i \in [1,L]
$$

**4、sentence attention**

和character级别的attention类似，使用一个segment级别的上下文向量$u_s$,来衡量一个segment在datagram的重要性。
$$
u_{i} = tanh(W_sh_{i}+b_s)
$$

$$
\alpha_i = \frac {exp(u_i^Tu_s)}{\sum_t(exp(u_t^Tu_s))}
$$

$$
d = \sum_{t} \alpha_{i}h_{i}
$$

##### 5、softmax

上面的$d$向量就是我们的到的最后的datagram表示，然后输入一个全连接的`softmax`层进行分类就ok了。

BSNN.py中包含了最重要的模型代码:

```python
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

def getSequenceLength(sequences):
    """
    :param sequences: 所有的segmetn长度，[a_size,b_size,c_size,,,]
    :return:每个segment进行padding前的实际大小
    """
    abs_sequences = tf.abs(sequences)
    # after padding data, max is 0
    abs_max_seq = tf.reduce_max(abs_sequences, reduction_indices=2)
    max_seq_sign = tf.sign(abs_max_seq)

    # sum is the real length
    real_len = tf.reduce_sum(max_seq_sign, reduction_indices=1)

    return tf.cast(real_len, tf.int32)

class BSNN(object):
    def __init__(self, max_sentence_num, max_sentence_length, num_classes, vocab_size,
                 embedding_size, learning_rate, decay_steps, decay_rate,
                 hidden_size, l2_lambda, grad_clip, is_training=False,
                 initializer=tf.random_normal_initializer(stddev=0.1)):
        
        self.vocab_size = vocab_size # 对应我的char数量 256
        self.max_sentence_num = max_sentence_num 
        self.max_sentence_length = max_sentence_length
        self.num_classes = num_classes # 分类数 3
        self.embedding_size = embedding_size # 对于char的emb_size=256 论文中初始化为one-hot
        self.hidden_size = hidden_size 
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.l2_lambda = l2_lambda # l2范数
        self.grad_clip = grad_clip
        self.initializer = initializer

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # placeholder
        self.input_x = tf.placeholder(tf.int32, [None, max_sentence_num, max_sentence_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        if not is_training:
            return

        word_embedding = self.word2vec() # 对于char表达，初始化为one-hot
        sen_vec = self.sen2vec(word_embedding) # 对应一个segment表达
        doc_vec = self.doc2vec(sen_vec) # 对应datagram表达

        self.logits = self.inference(doc_vec)
        self.loss_val = self.loss(self.input_y, self.logits)
        self.train_op = self.train()
        self.prediction = tf.argmax(self.logits, axis=1, name='prediction')
        self.pred_min = tf.reduce_min(self.prediction)
        self.pred_max = tf.reduce_max(self.prediction)
        self.pred_cnt = tf.bincount(tf.cast(self.prediction, dtype=tf.int32))
        self.label_cnt = tf.bincount(tf.cast(tf.argmax(self.input_y, axis=1), dtype=tf.int32))
        self.accuracy = self.accuracy(self.logits, self.input_y)

    def word2vec(self):
        with tf.name_scope('embedding'):
            self.embedding_mat = tf.Variable(tf.eye(self.vocab_size), name='embedding') # 构造256维度单位one-hot向量
            # [batch, sen_in_doc, wrd_in_sent, embedding_size]
            word_embedding = tf.nn.embedding_lookup(self.embedding_mat, self.input_x)
            return word_embedding

    def BidirectionalGRUEncoder(self, inputs, name):
        """
        双向GRU编码层，将一segment中的所有character或者一个datagram中的所有segment进行编码得到一个2xhidden_size的输出向量
        然后在输入inputs的shape是：
        input:[batch, max_time, embedding_size]
        output:[batch, max_time, 2*hidden_size]
        :return:
        """
        with tf.name_scope(name), tf.variable_scope(name, reuse=tf.AUTO_REUSE):  
            fw_gru_cell = rnn.GRUCell(self.hidden_size)
            bw_gru_cell = rnn.GRUCell(self.hidden_size)
            fw_gru_cell = rnn.DropoutWrapper(fw_gru_cell, output_keep_prob=self.dropout_keep_prob)
            bw_gru_cell = rnn.DropoutWrapper(bw_gru_cell, output_keep_prob=self.dropout_keep_prob)

            # fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            (fw_outputs, bw_outputs), (fw_outputs_state, bw_outputs_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_gru_cell, cell_bw=bw_gru_cell, inputs=inputs,
                sequence_length=getSequenceLength(inputs), dtype=tf.float32
            )
            # outputs的shape是[batch_size, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def AttentionLayer(self, inputs, name):
        """
        inputs是GRU层的输出
        inputs: [batch, max_time, 2*hidden_size]
        :return:
        """
        with tf.name_scope(name):
            context_weight = tf.Variable(tf.truncated_normal([self.hidden_size*2]), name='context_weight')

            # 使用单层MLP对GRU的输出进行编码，得到隐藏层表示
            # uit =tanh(Wwhit + bw)
            fc = layers.fully_connected(inputs, self.hidden_size*2, activation_fn=tf.nn.tanh)

            multiply = tf.multiply(fc, context_weight)
            reduce_sum = tf.reduce_sum(multiply, axis=2, keep_dims=True)
            # shape: [batch_size, max_time, 1]
            alpha = tf.nn.softmax(reduce_sum, dim=1)

            # shape: [batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output

    def sen2vec(self, word_embeded):
        with tf.name_scope('sen2vec'):
            """
            GRU的输入tensor是[batch_size, max_time,...]，在构造segment向量时max_time应该是每个segment的长度，
            所以这里将batch_size*sen_in_doc当做是batch_size，这样一来，每个GRU的cell处理的都是一个character的向量
            并最终将一segment中的所有character的向量融合（Attention）在一起形成segment向量
            """
            # shape：[batch_size * sen_in_doc, word_in_sent, embedding_size]
            word_embeded = tf.reshape(word_embeded, [-1, self.max_sentence_length, self.embedding_size])

            # shape: [batch_size * sen_in_doc, word_in_sent, hiddeng_size * 2]
            word_encoder = self.BidirectionalGRUEncoder(word_embeded, name='word_encoder')

            # shape: [batch_size * sen_in_doc, hidden_size * 2]
            sen_vec = self.AttentionLayer(word_encoder, name='word_attention')
            return sen_vec

    def doc2vec(self, sen_vec):
        with tf.name_scope('doc2vec'):
            """
            跟sen2vec类似，不过这里每个cell处理的是一个segment的向量，最后融合成为datagram的向量
            """
            sen_vec = tf.reshape(sen_vec, [-1, self.max_sentence_num, self.hidden_size*2])
            # shape: [batch_size，sen_in_doc, hidden_size * 2]
            doc_encoder = self.BidirectionalGRUEncoder(sen_vec, name='doc_encoder')
            # shape: [batch_size，hidden_size * 2]
            doc_vec = self.AttentionLayer(doc_encoder, name='doc_vec')
            return doc_vec

    def inference(self, doc_vec):
        with tf.name_scope('logits'):
            fc_out = layers.fully_connected(doc_vec, self.num_classes)
            return fc_out

    def accuracy(self, logits, input_y):
        with tf.name_scope('accuracy'):
            predict = tf.argmax(logits, axis=1, name='predict')
            label = tf.argmax(input_y, axis=1, name='label')
            acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))
            return acc

    def loss(self, input_y, logits):
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits)
            loss = tf.reduce_mean(losses)
            if self.l2_lambda >0:
                l2_loss = tf.add_n([tf.nn.l2_loss(cand_var) for cand_var in tf.trainable_variables() if 'bia' not in cand_var.name])
                loss += self.l2_lambda * l2_loss
            return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.decay_steps, self.decay_rate, staircase=True)
        # use grad_clip to hand exploding or vanishing gradients
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss_val)
        for idx, (grad, var) in enumerate(grads_and_vars):
            if grad is not None:
                grads_and_vars[idx] = (tf.clip_by_norm(grad, self.grad_clip), var)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return train_op
```

#### 训练参数

训练参数的设定可以看下面代码中configuration的内容，这里说一下论文中提到的几个参数，$\gamma = 1$，$\alpha$均等，segment长度设定为8，batch_size为30（与原文一致），epoch=10。

train.py

```python
import tensorflow as tf
import time
import os
import datetime
from BSNN import BSNN
from DataUtil import read_dataset

#configuration
tf.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.flags.DEFINE_integer("num_epochs",10,"embedding size")
tf.flags.DEFINE_integer("batch_size", 30, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.flags.DEFINE_integer("num_classes", 3, "number of classes")
tf.flags.DEFINE_integer("vocab_size", 256, "vocabulary size")
tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少

tf.flags.DEFINE_string("ckpt_dir","text_han_checkpoint/","checkpoint location for the model")
tf.flags.DEFINE_integer('num_checkpoints',5,'save checkpoints count')  

tf.flags.DEFINE_integer('max_sentence_num',100,'max sentence num in a doc')
tf.flags.DEFINE_integer('max_sentence_length',8,'max word count in a sentence')
tf.flags.DEFINE_integer("embedding_size",256,"embedding size")
tf.flags.DEFINE_integer('hidden_size',50,'cell output size')

tf.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")

tf.flags.DEFINE_integer("validate_every", 100, "Validate every validate_every epochs.") #每10轮做一次验证
tf.flags.DEFINE_float('validation_percentage',0.1,'validat data percentage in train data')

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float('grad_clip',2.0,'grad_clip') # 和类别数相关

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS

# load dataset
train_x, train_y, dev_x, dev_y = read_dataset()
print ("data load finished")

with tf.Graph().as_default():
    sess_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=sess_conf)

    with sess.as_default():
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run_BSNN', timestamp))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, FLAGS.ckpt_dir))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

        bsnn = BSNN(FLAGS.max_sentence_num, FLAGS.max_sentence_length, FLAGS.num_classes, FLAGS.vocab_size, FLAGS.embedding_size,
                  FLAGS.learning_rate, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.hidden_size, FLAGS.l2_reg_lambda,
                  FLAGS.grad_clip, FLAGS.is_training)
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {bsnn.input_x: x_batch,
                         bsnn.input_y: y_batch,
                         bsnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                         }
            tmp, step, loss, accuracy, label_cnt, pred_cnt, pred_min, pred_max = sess.run(
                [bsnn.train_op, bsnn.global_step, bsnn.loss_val, bsnn.accuracy, bsnn.label_cnt, bsnn.pred_cnt,
                 bsnn.pred_min, bsnn.pred_max,], feed_dict=feed_dict)

            print('train_label_cnt: ', label_cnt)
            print('train_min_max:', pred_min, pred_max)
            print('train_cnt: ', pred_cnt)
            time_str = datetime.datetime.now().isoformat()
            print("{}:step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return step

        def dev_step(dev_x, dev_y):
            feed_dict = {bsnn.input_x: dev_x,
                         bsnn.input_y: dev_y,
                         bsnn.dropout_keep_prob: 1.0
                         }
            step, loss, accuracy, label_cnt, pred_cnt = sess.run(
                [bsnn.global_step, bsnn.loss_val, bsnn.accuracy, bsnn.label_cnt, bsnn.pred_cnt,], feed_dict=feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("dev result: {}:step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


        for epoch in range(FLAGS.num_epochs):
            print('current epoch %s' % (epoch + 1))
            for i in range(0, 270, FLAGS.batch_size):
                x = train_x[i:i + FLAGS.batch_size]
                y = train_y[i:i + FLAGS.batch_size]
                train_step(x, y)
                cur_step = tf.train.global_step(sess, bsnn.global_step)

            print('\n')
            dev_step(dev_x, dev_y)
            path = saver.save(sess, checkpoint_prefix, global_step=epoch)
            print('Saved model checkpoint to {}\n'.format(path))
```

#### 结果

| epoch | 结果                        |
| ----- | --------------------------- |
| 1     | loss 1.17914, acc 0.206897  |
| 2     | loss 1.16526, acc 0.206897  |
| 3     | loss 1.17013, acc 0.206897  |
| 4     | loss 1.16233, acc 0.206897  |
| 5     | loss 1.14642, acc 0.448276  |
| 6     | loss 1.04893, acc 0.448276  |
| 7     | loss 0.898174, acc 0.448276 |
| 8     | loss 0.708184, acc 0.827586 |
| 9     | loss 0.616511, acc 0.827586 |
| 10    | loss 0.4711, acc 0.827586   |

部分输出细节见下：

```
current epoch 1
2021-02-05T23:28:26.090626:step 1, loss 1.18368, acc 0.233333
2021-02-05T23:28:26.623201:step 2, loss 1.17141, acc 0.366667
2021-02-05T23:28:27.112586:step 3, loss 1.17092, acc 0.2
2021-02-05T23:28:27.600048:step 4, loss 1.16999, acc 0.333333
2021-02-05T23:28:28.092669:step 5, loss 1.16888, acc 0.4
2021-02-05T23:28:28.593807:step 6, loss 1.1664, acc 0.4
2021-02-05T23:28:29.085813:step 7, loss 1.17391, acc 0.233333
2021-02-05T23:28:29.577284:step 8, loss 1.16082, acc 0.433333
2021-02-05T23:28:30.078963:step 9, loss 1.17725, acc 0.3


dev result: 2021-02-05T23:28:30.484341:step 9, loss 1.17914, acc 0.206897
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-0

current epoch 2
2021-02-05T23:28:31.313493:step 10, loss 1.14458, acc 0.466667
2021-02-05T23:28:31.801910:step 11, loss 1.16036, acc 0.366667
2021-02-05T23:28:32.340277:step 12, loss 1.20258, acc 0.2
2021-02-05T23:28:32.840851:step 13, loss 1.16389, acc 0.333333
2021-02-05T23:28:33.345806:step 14, loss 1.14916, acc 0.4
2021-02-05T23:28:33.865011:step 15, loss 1.15131, acc 0.4
2021-02-05T23:28:34.375607:step 16, loss 1.16807, acc 0.233333
2021-02-05T23:28:34.888499:step 17, loss 1.15018, acc 0.433333
2021-02-05T23:28:35.416810:step 18, loss 1.16337, acc 0.3


dev result: 2021-02-05T23:28:35.609295:step 18, loss 1.16526, acc 0.206897
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-1

current epoch 3
2021-02-05T23:28:36.307796:step 19, loss 1.14345, acc 0.466667
2021-02-05T23:28:36.807924:step 20, loss 1.15106, acc 0.366667
2021-02-05T23:28:37.413876:step 21, loss 1.18131, acc 0.2
2021-02-05T23:28:37.979364:step 22, loss 1.1481, acc 0.333333
2021-02-05T23:28:38.528895:step 23, loss 1.13485, acc 0.4
2021-02-05T23:28:39.053765:step 24, loss 1.14369, acc 0.4
2021-02-05T23:28:39.566839:step 25, loss 1.16878, acc 0.233333
2021-02-05T23:28:40.087222:step 26, loss 1.1341, acc 0.433333
2021-02-05T23:28:40.603841:step 27, loss 1.1617, acc 0.3


dev result: 2021-02-05T23:28:40.796326:step 27, loss 1.17013, acc 0.206897
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-2

current epoch 4
2021-02-05T23:28:41.602442:step 28, loss 1.1145, acc 0.466667
2021-02-05T23:28:42.144761:step 29, loss 1.14097, acc 0.366667
2021-02-05T23:28:42.693294:step 30, loss 1.20376, acc 0.2
2021-02-05T23:28:43.229859:step 31, loss 1.14151, acc 0.333333
2021-02-05T23:28:43.755452:step 32, loss 1.12037, acc 0.4
2021-02-05T23:28:44.263125:step 33, loss 1.13507, acc 0.4
2021-02-05T23:28:44.771254:step 34, loss 1.16051, acc 0.233333
2021-02-05T23:28:45.294753:step 35, loss 1.11591, acc 0.433333
2021-02-05T23:28:45.826330:step 36, loss 1.14527, acc 0.3


dev result: 2021-02-05T23:28:46.004852:step 36, loss 1.16233, acc 0.206897
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-3

current epoch 5
2021-02-05T23:28:46.711379:step 37, loss 1.10953, acc 0.466667
2021-02-05T23:28:47.210262:step 38, loss 1.13091, acc 0.366667
2021-02-05T23:28:47.773246:step 39, loss 1.18087, acc 0.2
2021-02-05T23:28:48.286123:step 40, loss 1.13255, acc 0.333333
2021-02-05T23:28:48.789008:step 41, loss 1.11187, acc 0.4
2021-02-05T23:28:49.302116:step 42, loss 1.13908, acc 0.4
2021-02-05T23:28:49.805403:step 43, loss 1.1431, acc 0.266667
2021-02-05T23:28:50.314138:step 44, loss 1.11676, acc 0.433333
2021-02-05T23:28:50.827770:step 45, loss 1.12081, acc 0.366667


dev result: 2021-02-05T23:28:51.000757:step 45, loss 1.14642, acc 0.448276
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-4

current epoch 6
2021-02-05T23:28:51.710229:step 46, loss 1.1051, acc 0.566667
2021-02-05T23:28:52.197731:step 47, loss 1.09202, acc 0.5
2021-02-05T23:28:52.747259:step 48, loss 1.15978, acc 0.333333
2021-02-05T23:28:53.266820:step 49, loss 1.09888, acc 0.5
2021-02-05T23:28:53.778483:step 50, loss 1.04828, acc 0.6
2021-02-05T23:28:54.300956:step 51, loss 1.10639, acc 0.5
2021-02-05T23:28:54.807825:step 52, loss 1.06262, acc 0.5
2021-02-05T23:28:55.310532:step 53, loss 1.03075, acc 0.5
2021-02-05T23:28:55.833391:step 54, loss 1.02559, acc 0.466667


dev result: 2021-02-05T23:28:56.026642:step 54, loss 1.04893, acc 0.448276
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-5

current epoch 7
2021-02-05T23:28:56.747713:step 55, loss 0.996561, acc 0.566667
2021-02-05T23:28:57.233785:step 56, loss 0.973121, acc 0.5
2021-02-05T23:28:57.788470:step 57, loss 1.10553, acc 0.333333
2021-02-05T23:28:58.301127:step 58, loss 0.873215, acc 0.5
2021-02-05T23:28:58.816781:step 59, loss 0.839209, acc 0.633333
2021-02-05T23:28:59.331319:step 60, loss 0.928375, acc 0.533333
2021-02-05T23:28:59.847088:step 61, loss 0.837925, acc 0.5
2021-02-05T23:29:00.344677:step 62, loss 0.871867, acc 0.533333
2021-02-05T23:29:00.813062:step 63, loss 0.89659, acc 0.5


dev result: 2021-02-05T23:29:00.985665:step 63, loss 0.898174, acc 0.448276
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-6

current epoch 8
2021-02-05T23:29:01.699028:step 64, loss 0.867806, acc 0.633333
2021-02-05T23:29:02.186859:step 65, loss 0.826654, acc 0.633333
2021-02-05T23:29:02.737684:step 66, loss 1.01797, acc 0.433333
2021-02-05T23:29:03.243936:step 67, loss 0.765, acc 0.666667
2021-02-05T23:29:03.746920:step 68, loss 0.721655, acc 0.8
2021-02-05T23:29:04.245246:step 69, loss 0.937242, acc 0.8
2021-02-05T23:29:04.742633:step 70, loss 0.662761, acc 0.766667
2021-02-05T23:29:05.246319:step 71, loss 0.838433, acc 0.8
2021-02-05T23:29:05.758371:step 72, loss 0.69352, acc 0.833333


dev result: 2021-02-05T23:29:05.941858:step 72, loss 0.708184, acc 0.827586
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-7

current epoch 9
2021-02-05T23:29:06.681740:step 73, loss 0.615287, acc 0.833333
2021-02-05T23:29:07.155841:step 74, loss 0.634121, acc 0.833333
2021-02-05T23:29:07.706681:step 75, loss 0.863259, acc 0.7
2021-02-05T23:29:08.233564:step 76, loss 0.527865, acc 0.866667
2021-02-05T23:29:08.728556:step 77, loss 0.626868, acc 0.8
2021-02-05T23:29:09.239092:step 78, loss 0.655192, acc 0.833333
2021-02-05T23:29:09.747197:step 79, loss 0.518165, acc 0.833333
2021-02-05T23:29:10.261518:step 80, loss 0.557925, acc 0.866667
2021-02-05T23:29:10.763176:step 81, loss 0.600361, acc 0.833333


dev result: 2021-02-05T23:29:10.931726:step 81, loss 0.616511, acc 0.827586
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-8

current epoch 10
2021-02-05T23:29:11.619203:step 82, loss 0.579097, acc 0.833333
2021-02-05T23:29:12.099981:step 83, loss 0.531345, acc 0.833333
2021-02-05T23:29:12.636098:step 84, loss 0.777414, acc 0.7
2021-02-05T23:29:13.135171:step 85, loss 0.438025, acc 0.866667
2021-02-05T23:29:13.644192:step 86, loss 0.477813, acc 0.833333
2021-02-05T23:29:14.140262:step 87, loss 0.545707, acc 0.833333
2021-02-05T23:29:14.631534:step 88, loss 0.440086, acc 0.833333
2021-02-05T23:29:15.129455:step 89, loss 0.427449, acc 0.866667
2021-02-05T23:29:15.646563:step 90, loss 0.474299, acc 0.833333


dev result: 2021-02-05T23:29:15.826884:step 90, loss 0.4711, acc 0.827586
Saved model checkpoint to C:\Users\Garli\Desktop\BSNN\run_BSNN\1612538901\text_han_checkpoint\model-9
```

#### 不足

- 数据集收集时间比较仓促，因此样本不够满意，数据量由于电脑性能限制也比较少
- Focal Loss针对我的数据特点使用$\alpha=[[1],[1],[1]],\gamma=1$等同于`softmax_cross_entropy_with_logits`，效果不错，但没有做详尽的对比实验较难看出它的优势
- embedding的维度设置的256，但感觉数据表示比较稀疏
- 对于character的这一层的训练，我的直觉上来说是远不如单词训练之于句子的意义性，因为相同的byte可能不具有很强的关联性，如果直接将segment的数字表示作为输入，减少一层不知道效果是否会退化很多

### 参考链接

https://ieeexplore.ieee.org/abstract/document/8624128

https://www.aclweb.org/anthology/N16-1174.pdf

https://zihuaweng.github.io/2018/04/01/loss/

https://zhuanlan.zhihu.com/p/80594704

https://zhuanlan.zhihu.com/p/35571412