from node2vec import Node2Vec
import networkx as nx
import pandas as pd
import numpy as np
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

kg = pd.read_csv("./crf/syndrome9.csv")
G = nx.from_pandas_edgelist(kg, 'syndrome', 'symptom', edge_attr=None, create_using=nx.Graph())

patient1 = pd.read_excel('./crf/sk1.xlsx')
patient2 = pd.read_excel('./crf/sk2.xlsx')
patient3 = pd.read_excel('./crf/sk3.xlsx')

name = list(patient1.columns)[8: ]
print('患者症状名称：', name)

kg_name = list(set(list(kg['syndrome']) + list(kg['symptom'])))
print('知识图谱节点名称：', kg_name)

# 空值处理
def padnan(metrix):
    for i in range(metrix.shape[0]):
        for j in range(metrix.shape[1]):
            if np.isnan(metrix[i, j]):
                metrix[i, j] = 0
    return metrix

patient1 = padnan(np.array(patient1))
patient2 = padnan(np.array(patient2))
patient3 = padnan(np.array(patient3))

patient = np.concatenate((patient1, patient2, patient3), axis = 0)
print('患者特征矩阵：', patient.shape)

patient = list(patient)
# randnum = random.randint(0, 100)
randnum = 2 # 2,4,7,11
random.seed(randnum)
random.shuffle(patient)
patient = np.array(patient)

# 划分训练集测试集
train_pat = patient[0: int(len(patient)*0.6)]
val_pat = patient[int(len(patient)*0.6): int(len(patient)*0.8)]
test_pat = patient[int(len(patient)*0.8): ]

# 训练集样本增强
train_pat_aug = []
for i in range(len(train_pat[:, 0])):
    if train_pat[i, 0] == 0:
        train_pat_aug.append(train_pat[i])
    if train_pat[i, 0] == 2:
        train_pat_aug.append(train_pat[i])
    if train_pat[i, 0] == 3:
        train_pat_aug.append(train_pat[i])

# 验证集样本增强
val_pat_aug = []
for i in range(len(val_pat[:, 0])):
    if val_pat[i, 0] == 0:
        val_pat_aug.append(val_pat[i])
    if val_pat[i, 0] == 2:
        val_pat_aug.append(val_pat[i])
    if val_pat[i, 0] == 3:
        val_pat_aug.append(val_pat[i])

train_pat_aug = np.array(1 * train_pat_aug) # 增强1次
train_pat = np.concatenate((train_pat, train_pat_aug), axis = 0)

val_pat_aug = np.array(1 * val_pat_aug) # 增强1次
val_pat = np.concatenate((val_pat, val_pat_aug), axis = 0)

patient = np.concatenate((train_pat, val_pat, test_pat), axis = 0)

# onehot label转整数
label = patient[:, 0:4]
patient_label = []
for i in range(len(label)):
    patient_label.append(list(label[i]).index(max(label[i])))

patient_feat = patient[:, 8:]
# print(patient_feat)

# 患者症状矩阵转患者症状名称
patient_feat_name = []
for i in range(len(patient_feat)):
    temp = []
    for j in range(len(patient_feat[i])):
        if patient_feat[i, j] != 0:
            temp.append(name[j])
    patient_feat_name.append(temp)

print(patient_feat_name)

n2v = Node2Vec(G, dimensions=16, walk_length=10, num_walks=50, workers=1, seed=2)
# n2v.random.seed()
model = n2v.fit(window=5, min_count=1, batch_words=4)

print(model.wv['气虚血瘀'])

embeddings = {}
for i in range(len(kg_name)):
    embeddings[kg_name[i]] = model.wv[kg_name[i]]

patient_embed = []
for i in range(len(patient_feat_name)):
    temp = []
    for j in range(len(patient_feat_name[i])):
        if patient_feat_name[i][j] in embeddings.keys():
            temp.append(embeddings[patient_feat_name[i][j]].reshape(1, -1))
    patient_embed.append(np.array(temp).mean(0))

patient_embed = np.array(patient_embed).squeeze()

print(patient_embed.shape)

train_X = patient_embed[: 439]
train_Y = patient_label[: 439]

test_X = patient_embed[439 + 142:]
test_Y = patient_label[439 + 142:]

mlp = MLPClassifier(hidden_layer_sizes=(), max_iter=150, random_state=2)

mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
print('准确率：', accuracy_score(pred, test_Y))

conf_mat = confusion_matrix(test_Y, pred)
print('混淆矩阵：', conf_mat)

print(classification_report(test_Y, pred))