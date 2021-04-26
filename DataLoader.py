import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np
import networkx as nx

np.random.seed(0) #2
torch.manual_seed(0) #2


class GraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='knowledge_graph')

    def process(self):
        nodes_labels = pd.read_csv('./interdata/patient_node_label.csv')
        edges_data = pd.read_csv('./interdata/graph.csv')
        # node_features1 = np.load('./data/glove.npy')
        # node_features2 = np.load('./data/demographic.npy')
        # node_features = np.concatenate((node_features1[:135], node_features2), axis=0)

        # node_features = np.load('./data/glove_8.npy')
        node_features = np.random.normal(0, 0.01, size=(828, 8))
        # node_features = torch.randn(702, 8)

        node_features = torch.tensor(torch.from_numpy(node_features), dtype=torch.float)
        node_labels = torch.from_numpy(nodes_labels['label'].to_numpy())
        # edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())

        g = dgl.graph((edges_src, edges_dst), num_nodes=828)
        # g = dgl.DGLGraph((edges_src, edges_dst), num_nodes=1884, multigraph=True)
        self.graph = dgl.to_bidirected(g)

        # G = nx.from_pandas_edgelist(edges_data, 'src', 'dst', edge_attr=None, create_using=nx.Graph())
        # self.graph = G

        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        # self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.

        n_nodes = node_labels.shape[0]
        # n_train = int((n_nodes-135) * 0.6)
        # n_val = int((n_nodes-135) * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[139: 139 + 439] = True
        # train_mask[0: 9] = True # 证型节点也做分类
        val_mask[139 + 439: 139 + 439 + 142] = True
        test_mask[139 + 439 + 142:] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

# g= GraphDataset()
# print(g[0])