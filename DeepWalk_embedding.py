import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


random.seed(2)

kg = pd.read_csv("./data/syndrome9.csv")
G = nx.from_pandas_edgelist(kg, 'syndrome', 'symptom', edge_attr=None, create_using=nx.Graph())

patient1 = pd.read_excel('./data/crf1_4_10.xlsx')
patient2 = pd.read_excel('./data/crf2_4_10.xlsx')

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

patient = np.concatenate((patient1, patient2), axis = 0)
print('患者特征矩阵：', patient.shape)

patient = list(patient)
# randnum = random.randint(0, 100)
randnum = 2 # 2,4,7,11
random.seed(randnum)
random.shuffle(patient)
patient = np.array(patient)

# 划分训练集测试集
train_patient = patient[0: int(len(patient)*0.6)]
val_patient = patient[int(len(patient)*0.6): int(len(patient)*0.8)]
test_patient = patient[int(len(patient)*0.8): ]

# 负类样本增强
train_patient_neg = []
for i in range(len(train_patient[:, 0:4])):
    if list(train_patient[:, 0:4][i]).index(max(train_patient[:, 0:4][i])) != 1:
        train_patient_neg.append(train_patient[i])

train_patient_neg = np.array(1 * train_patient_neg) # 增强1次
train_patient = np.concatenate((train_patient, train_patient_neg), axis = 0)

print('训练集样本数：', len(train_patient))
print('验证集样本数：', len(val_patient))
print('测试集样本数：', len(test_patient))

patient = np.concatenate((train_patient, val_patient, test_patient), axis = 0)

# onehot label转整数
label = patient[:, 0:4]
patient_label = []
for i in range(len(label)):
    patient_label.append(list(label[i]).index(max(label[i])))

for i in range(len(patient_label)):
    if patient_label[i] != 1:
        patient_label[i] = 0

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

def get_randomwalk(node, path_length):
    random_walk = [node]

    for i in range(path_length - 1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node

    return random_walk

# print(get_randomwalk('space exploration', 10))

# get list of all nodes from the graph
all_nodes = list(G.nodes())

random_walks = []
for n in tqdm(all_nodes):
    for i in range(5):
        random_walks.append(get_randomwalk(n, 10))

# count of sequences
print(len(random_walks))
print('random_walks list:', random_walks)

from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')

model = Word2Vec(size=16, window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007, workers = 1,
                 seed = 2)

model.build_vocab(random_walks, progress_per=2)

model.train(random_walks, total_examples = model.corpus_count, epochs=50, report_delay=1)

# terms = list(G.nodes)

# print('***', model.wv.vectors)
print('气虚血瘀', model.wv['气虚血瘀'])

embeddings = {}
for i in range(len(kg_name)):
    embeddings[kg_name[i]] = model.wv[kg_name[i]]

# print(embeddings)

# print(embeddings['气虚血瘀'])

patient_embed = []
for i in range(len(patient_feat_name)):
    temp = []
    for j in range(len(patient_feat_name[i])):
        if patient_feat_name[i][j] in embeddings.keys():
            temp.append(embeddings[patient_feat_name[i][j]].reshape(1, -1))
    patient_embed.append(np.array(temp).mean(0))

patient_embed = np.array(patient_embed).squeeze()

print(patient_embed.shape)

train_X = patient_embed[: 360]
train_Y = patient_label[: 360]

test_X = patient_embed[360 + 89:]
test_Y = patient_label[360 + 89:]

mlp = MLPClassifier(hidden_layer_sizes=(), max_iter=150, random_state=2)

mlp.fit(train_X, train_Y)

pred = mlp.predict(test_X)
print('准确率：', accuracy_score(pred, test_Y))

conf_mat = confusion_matrix(test_Y, pred)
print('混淆矩阵：', conf_mat)

print(classification_report(test_Y, pred))