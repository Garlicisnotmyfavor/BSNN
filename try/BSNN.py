import torch
import torch.nn as nn
import torch.optim as optim
import csv
import torch.utils.data as data  
import numpy as np
import time
import math
import pre as pre
import torch.nn.functional as F

# 超参数
HIDDEN_SIZE = 10 # 隐藏层
BATCH_SIZE = 10
N_LAYER = 2 # RNN的层数
N_EPOCHS = 10 # train的轮数
N_CHARS = 256 # 这个就是要构造的字典的长度
USE_GPU = False
SEGMENT_LEN = 8
EMBEDDING_DIM = 256
LEARNING_RATE = 1e-3  

