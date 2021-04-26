import pandas as pd
import numpy as np
import random
from docx import Document
from sklearn.preprocessing import MinMaxScaler

# 知识库2
document = Document("./crf/knowledge_graph.docx")
all_paragraphs = document.paragraphs

kg2 = []
for paragraph in all_paragraphs:
    if paragraph.text != '':
        # print(paragraph.text.split('#')[2].split('，'))
        for i in range(len(paragraph.text.split('#')[2].split('，'))):
            temp = []
            temp.append(paragraph.text.split('#')[0])
            temp.append(paragraph.text.split('#')[2].split('，')[i])
            kg2.append(temp)

# print(kg2)

save = pd.DataFrame([['syndrome', 'symptom']] + kg2)
save.to_csv('./crf/syndrome9.csv', index = False, sep=',', header = None, encoding = 'utf_8_sig')


kg2 = np.array(kg2)

# 证型编号
syndrome = {}
count = 0
for i in range(len(kg2[:, 0])):
    if kg2[:, 0][i] not in syndrome:
        syndrome[kg2[:, 0][i]] = count
        count = count + 1

print(syndrome)


# 知识库症状编号
print('症状起始编号：', count)
symptom = {}
for i in range(len(kg2[:, 1])):
    if kg2[:, 1][i] not in symptom:
        symptom[kg2[:, 1][i]] = count
        count = count + 1

patient1 = pd.read_excel('./crf/sk1.xlsx')
patient2 = pd.read_excel('./crf/sk2.xlsx')
patient3 = pd.read_excel('./crf/sk3.xlsx')

name = list(patient1.columns)
print('name: ', name)
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

print('中风数据维度：', patient.shape)
print('风痰阻络：', sum(patient[:, 0]))
print('气虚血瘀：', sum(patient[:, 1]))
print('痰热腑实：', sum(patient[:, 2]))
print('阴虚动风：', sum(patient[:, 3]))


# 处理证型标签
sk_label = []
for i in range(len(patient[:, 0:4])):
    if list(patient[:, 0:4][i]).index(max(patient[:, 0:4][i])) == 0:
        sk_label.append(0)
    if list(patient[:, 0:4][i]).index(max(patient[:, 0:4][i])) == 1:
        sk_label.append(1)
    if list(patient[:, 0:4][i]).index(max(patient[:, 0:4][i])) == 2:
        sk_label.append(2)
    if list(patient[:, 0:4][i]).index(max(patient[:, 0:4][i])) == 3:
        sk_label.append(3)

patient = np.concatenate((np.array(sk_label).reshape(-1, 1), patient[:, 4:]), axis =1)


# 归一化
# 年龄、性别、身高、体重
mm = MinMaxScaler()
feat1 = mm.fit_transform(patient[:, 1].reshape(-1, 1))
feat3 = mm.fit_transform(patient[:, 3].reshape(-1, 1))
feat4 = mm.fit_transform(patient[:, 4].reshape(-1, 1))

patient = np.concatenate((patient[:, 0].reshape(-1, 1), feat1, patient[:, 2].reshape(-1, 1), feat3, feat4, patient[:, 5:]), axis = 1)

# print(patient[2])

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

# 训练集样本增强
train_pat_aug = []
for i in range(len(train_patient[:, 0])):
    if train_patient[i, 0] == 0:
        train_pat_aug.append(train_patient[i])
    if train_patient[i, 0] == 2:
        train_pat_aug.append(train_patient[i])
    if train_patient[i, 0] == 3:
        train_pat_aug.append(train_patient[i])

# 验证集样本增强
val_pat_aug = []
for i in range(len(val_patient[:, 0])):
    if val_patient[i, 0] == 0:
        val_pat_aug.append(val_patient[i])
    if val_patient[i, 0] == 2:
        val_pat_aug.append(val_patient[i])
    if val_patient[i, 0] == 3:
        val_pat_aug.append(val_patient[i])

train_pat_aug = np.array(1 * train_pat_aug) # 增强1次
train_patient = np.concatenate((train_patient, train_pat_aug), axis = 0)

val_pat_aug = np.array(1 * val_pat_aug) # 增强1次
val_patient = np.concatenate((val_patient, val_pat_aug), axis = 0)

print('训练集样本数：', len(train_patient))
print('验证集样本数：', len(val_patient))
print('测试集样本数：', len(test_patient))

patient = np.concatenate((train_patient, val_patient, test_patient), axis = 0)

# 患者症状编号
print('患者症状起始编号：', count)
for i in range(len(name[8:])): # 症状从序号8开始
    if name[8:][i] not in symptom:
        symptom[name[8:][i]] = count
        count = count + 1
print('syndrome: ', syndrome)
print('symptom: ', symptom)

node_order = dict(syndrome, **symptom)
print('node_order: ', node_order)


# 取出患者的症状
patient_edge = []
for i in range(len(patient)):
    temp = []
    for j in range(5, len(patient[i])): # 1-4体格特征不算
        if patient[i, j] == 1:
            temp.append(name[j+3])
    patient_edge.append(temp)

# print(len(patient_edge))


# patient 分配节点名称 patient0, patient1
patient_dict = {}
for i in range(len(patient_edge)):
    patient_dict['patient' + str(i)] = patient_edge[i]
print(patient_dict)


# 患者编号
print('患者起始编号：', count)
patient_index = count
patient_order = {}
for i in range(len(list(patient_dict.keys()))):
    if list(patient_dict.keys())[i] not in patient_order:
        patient_order[list(patient_dict.keys())[i]] = count
        count = count + 1
print(patient_order)

node_order = dict(node_order, **patient_order)
# print('节点编号：', node_order)

# 构建症状与患者节点的边
patient_node = []
# patient_node.append(['src', 'dst'])
for key, value in patient_dict.items():
    for i in range(len(value)):
        temp = []
        temp.append(value[i])
        temp.append(key)

        patient_node.append(temp)

# print(patient_node)

print('节点总数：', len(node_order))


# label 不需要构建到图上
node_label = []
node_label.append(['id', 'label'])
for i in range(patient_index): # patient序号前的节点的标签记为-1，先不考虑证型节点分类
    temp = []
    temp.append(i)
    temp.append(-1)
    node_label.append(temp)

for i in range(len(patient[:, 0])):
    temp = []
    temp.append(i + patient_index)
    temp.append(patient[:, 0][i])
    node_label.append(temp)

node_label = pd.DataFrame(node_label)
node_label.to_csv('./interdata/patient_node_label.csv', index = False, sep=',', header = None, encoding = 'utf_8_sig')

print('patient_node: ', patient_node)


kg_list = []
for i in range(len(kg2)):
    kg_list.append(list(kg2[i]))
print('kg_list: ', kg_list)

# 转序号
for i in range(len(kg_list)):
    for j in range(len(kg_list[i])):
        kg_list[i][j] = node_order[kg_list[i][j]]

for i in range(len(patient_node)):
    for j in range(len(patient_node[i])):
        patient_node[i][j] = node_order[patient_node[i][j]]

# 边数量
graph = kg_list + patient_node
print(len(graph))

# 去除重复边
graph_tuple = []
for i in range(len(graph)):
    graph_tuple.append(tuple(graph[i]))

graph_no_para = []
for i in set(graph_tuple):
    graph_no_para.append(list(i))
print(len(graph_no_para))

save = pd.DataFrame([['src', 'dst']] + graph_no_para)
save.to_csv('./interdata/graph.csv', index = False, sep=',', header = None, encoding = 'utf_8_sig')
