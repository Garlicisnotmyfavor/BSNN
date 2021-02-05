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