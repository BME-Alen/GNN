import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

def padnan(metrix):
    for i in range(metrix.shape[0]):
        for j in range(metrix.shape[1]):
            if np.isnan(metrix[i, j]):
                metrix[i, j] = 0
    return metrix

# [0:4]证型, [4:8]体格:年龄、性别、身高、体重
sk1 = pd.read_excel('./crf/sk1.xlsx')
sk2 = pd.read_excel('./crf/sk2.xlsx')
sk3 = pd.read_excel('./crf/sk3.xlsx')
sk_name = list(sk1.columns)
sk_mat = np.concatenate((np.array(sk1), np.array(sk2), np.array(sk3)), axis = 0)
# 填充nan
sk_mat = padnan(sk_mat)
print('中风数据维度：', sk_mat.shape)
print('风痰阻络：', sum(sk_mat[:, 0]))
print('气虚血瘀：', sum(sk_mat[:, 1]))
print('痰热腑实：', sum(sk_mat[:, 2]))
print('阴虚动风：', sum(sk_mat[:, 3]))

# 处理证型标签
sk_label = []
for i in range(len(sk_mat[:, 0:4])):
    if list(sk_mat[:, 0:4][i]).index(max(sk_mat[:, 0:4][i])) == 0:
        sk_label.append(0)
    if list(sk_mat[:, 0:4][i]).index(max(sk_mat[:, 0:4][i])) == 1:
        sk_label.append(1)
    if list(sk_mat[:, 0:4][i]).index(max(sk_mat[:, 0:4][i])) == 2:
        sk_label.append(2)
    if list(sk_mat[:, 0:4][i]).index(max(sk_mat[:, 0:4][i])) == 3:
        sk_label.append(3)

sk_mat = np.concatenate((np.array(sk_label).reshape(-1, 1), sk_mat[:, 4:]), axis =1)

print('中风数据维度：', sk_mat.shape)

# 归一化
# 年龄、性别、身高、体重
mm = MinMaxScaler()
feat1 = mm.fit_transform(sk_mat[:, 1].reshape(-1, 1))
feat3 = mm.fit_transform(sk_mat[:, 3].reshape(-1, 1))
feat4 = mm.fit_transform(sk_mat[:, 4].reshape(-1, 1))

sk_mat = np.concatenate((sk_mat[:, 0].reshape(-1, 1), feat1, sk_mat[:, 2].reshape(-1, 1), feat3, feat4, sk_mat[:, 5:]), axis = 1)

sk_mat = list(sk_mat)
randnum = 2
random.seed(randnum)
random.shuffle(sk_mat)
sk_mat = np.array(sk_mat)

train_pat = sk_mat[0: int(len(sk_mat)*0.6)]
val_pat = sk_mat[int(len(sk_mat)*0.6): int(len(sk_mat)*0.8)]
test_pat = sk_mat[int(len(sk_mat)*0.8):]

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




from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

mlp = MLPClassifier(hidden_layer_sizes=(256, ), max_iter=150, random_state=1)

mlp.fit(train_pat[:, 1:], train_pat[:, 0])

pred = mlp.predict(test_pat[:, 1:])
print('准确率：', accuracy_score(pred, test_pat[:, 0]))

conf_mat = confusion_matrix(test_pat[:, 0], pred)
print('混淆矩阵：', conf_mat)

print(classification_report(test_pat[:, 0], pred))

'''
prob = mlp.predict_proba(test_pat[:, 1:])

fpr, tpr, thresholds = roc_curve(test_pat[:, 0], prob[:, 1])
AUC = auc(fpr, tpr)

plt.figure()
# plt.title('ROC CURVE (AUC={:.2f})'.format(AUC))
plt.xlabel('False Positive Rate', size = 14)
plt.ylabel('True Positive Rate', size = 14)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.plot(fpr, tpr, color='g', label='MLP {:.2f}'.format(AUC))
plt.plot([0, 1], [0, 1], color='m', linestyle='--')

plt.legend(loc='lower right')
plt.show()
'''