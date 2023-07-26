import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import ChebConv, NNConv, DeepGCNLayer, GATConv, DenseGCNConv, GCNConv, GraphConv
from torch_geometric.data import Data, DataLoader


import warnings
warnings.filterwarnings("ignore")


df_features = pd.read_csv('elliptic_txs_features.csv',header=None)
df_edges = pd.read_csv("elliptic_txs_edgelist.csv")
df_classes =  pd.read_csv("elliptic_txs_classes.csv")
#Extract data from folders and assign a label to each class (1 for presence, 0 for absence, 2 for unknown)
df_classes['class'] = df_classes['class'].map({'unknown': 2, '1':1, '2':0})
# merging dataframes
df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
df_merge = df_merge.sort_values(0).reset_index(drop=True)
classified = df_merge.loc[df_merge['class'].loc[df_merge['class']!=2].index].drop('txId', axis=1)
unclassified = df_merge.loc[df_merge['class'].loc[df_merge['class']==2].index].drop('txId', axis=1)
classified_edges = df_edges.loc[df_edges['txId1'].isin(classified[0]) & df_edges['txId2'].isin(classified[0])]
unclassifed_edges = df_edges.loc[df_edges['txId1'].isin(unclassified[0]) | df_edges['txId2'].isin(unclassified[0])]
del df_features, df_classes
nodes = df_merge[0].values
map_id = {j:i for i,j in enumerate(nodes)} # mapping nodes to indexes
edges = df_edges.copy()
edges.txId1 = edges.txId1.map(map_id)
edges.txId2 = edges.txId2.map(map_id)
edges = edges.astype(int)
edge_index = np.array(edges.values).T
edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
#We assign weights, defaulting to 1
weights = torch.tensor([1]* edge_index.shape[1] , dtype=torch.double)
node_features = df_merge.drop(['txId'], axis=1).copy()
node_features[0] = node_features[0].map(map_id)
classified_idx = node_features['class'].loc[node_features['class']!=2].index
unclassified_idx = node_features['class'].loc[node_features['class']==2].index
node_features['class'] = node_features['class'].replace(2, 0)
labels = node_features['class'].values
node_features = torch.tensor(np.array(node_features.drop([0, 'class', 1], axis=1).values, dtype=np.double), dtype=torch.double)
data_train = Data(x=node_features, edge_index=edge_index, edge_attr=weights,
                               y=torch.tensor(labels, dtype=torch.double))
y_train = labels[classified_idx]
#We split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid, train_idx, valid_idx = train_test_split(node_features[classified_idx], y_train, classified_idx, test_size=0.15, random_state=42, stratify=y_train)
data_train.y[classified_idx].sum()
import gc
gc.collect()

import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #We define GNN layers based on the test results, using the best-performing parameters
        self.conv1 = GCNConv(165, 360)
        self.conv4 = GCNConv(360, 1)

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv4(x, edge_index)

        return F.sigmoid(x)
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
#Since there are significantly more instances labeled as class 0 than class 1, we assign higher weights to class 1
weight_tensor = torch.tensor(class_weights, dtype=torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
model.double()
data_train = data_train.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
s=[]
for i in range(0,len(y_train)):
    if y_train[i]==0:
        s.append(class_weights[0])
    else:
        s.append(class_weights[1])
weight_tensor = torch.tensor(s, dtype=torch.float)
criterion = torch.nn.BCELoss(weight=weight_tensor)
model.train()
for epoch in range(15):
    optimizer.zero_grad()
    out = model(data_train)
    out = out.reshape((data_train.x.shape[0]))
    loss = criterion(out[train_idx], data_train.y[train_idx])
    #We perform backpropagation only on the training set with indices 1 and 0 during the operation of backpropagation.
    loss.backward()
    optimizer.step()
    preds = model(data_train)
    preds = preds.detach().cpu().numpy()
    v = []
    k0 = 0
    for i in preds[valid_idx]:
        if i[0] < 0.5:
            v.append(0)
        else:
            v.append(1)
            k0 = k0 + 1
    print(k0)
    k0 = 0
    k1 = 1
    a0 = 0
    a1 = 0
    d = data_train.y[valid_idx].numpy()
    v = np.array(v)
    #Now, on each epoch, we evaluate the quality metric for both classes
    for i in range(len(d)):
        if d[i] == 1:
            a1 = a1 + 1
        if d[i] == 0:
            a0 = a0 + 1
        if d[i] == 0 and v[i] == d[i]:
            k0 = k0 + 1
        if d[i] == 1 and v[i] == d[i]:
            k1 = k1 + 1
    print(k0 / a0)
    print(k1 / a1)
    print(epoch)
