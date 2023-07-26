import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from collections import Counter

def most_frequent_numbers(lst):
    counter = Counter(lst)
    max_count = max(counter.values())
    most_frequent = [num for num, count in counter.items() if count == max_count]
    return most_frequent


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
#We create a GAT (Graph Attention Network) model

class GATModel(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GATModel, self).__init__()
        self.num_heads = num_heads
        self.gat = GATConv(in_features, 2, heads=num_heads).double()
        self.fc = nn.Linear(out_features * num_heads, 2).double()

    def forward(self, data):
        x, edge_index, edge_attr = data.x.double(), data.edge_index.long(), data.edge_attr.double()
        x = self.gat(x, edge_index, edge_attr)
        x = self.fc(x)
        return x
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
#Since there are significantly more instances labeled as class 0 than class 1, we assign higher weights to class 1
weight_tensor = torch.tensor(class_weights, dtype=torch.double)
q=[]
q1=[]
for s in range(0,200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_features = node_features.shape[1]
    out_features = 120
    num_heads = 8
    model = GATModel(in_features, 2, num_heads).to(device)
    data_train = data_train.to(device)
    #We perform backpropagation only on the training set with indices 1 and 0 during the operation of backpropagation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    y_train = torch.tensor(y_train, dtype=torch.long)
    for epoch in range(40):
        model.train()
        optimizer.zero_grad()
        logits = model(data_train)
        loss = criterion(logits[train_idx], y_train)
        loss.backward()
        optimizer.step()
        preds = model(data_train)
        preds = preds.detach().cpu().numpy()
        d = data_train.y[valid_idx].numpy()
        v = []
        v1=[]
        for i in preds[valid_idx]:
            v1.append(i.max())
            if i[0] == i.max():
                v.append(0)
            if i[1] == i.max():
                v.append(1)
        k0 = 0
        k1 = 1
        a0 = 0
        a1 = 0
        d = data_train.y[valid_idx].numpy()
        v = np.array(v)
        v1 = np.array(v1)
        for i in range(len(d)):
            if d[i] == 1:
                a1 = a1 + 1
            if d[i] == 0:
                a0 = a0 + 1
            if d[i] == 0 and v[i] == d[i]:
                k0 = k0 + 1
            if d[i] == 1 and v[i] == d[i]:
                k1 = k1 + 1
        l = 0
        for i in range(len(d)):
            if v[i] == d[i]:
                l = l + 1
        #Now, on each epoch, we evaluate the quality metric for both classes.
        print(k0 / a0)
        print(k1 / a1)
        print("acc:", l / len(d))
        print("========================")
        print(epoch)

        model.eval()
    t=str(s)
    for dfsgfsd in range(0,10):
        t=t+t
    print(t)
    #Now we save the probabilities and class labels for voting. We will have a total of 200 neural networks
    q.append(v)
    q1.append(v1)
d1=[]
#Now we conduct voting. Only the neural networks that are at least 80% confident in their prediction are allowed to participate in the voting. If there are no such networks, we select the top 10 most confident neural networks and proceed with the voting
for i in range(0,len(q[0])):
    f=[]
    f1=[]
    for j in range(0,len(q)):
        if q1[j][i]>=0.8:
            f.append(q[j][i])
        f1.append(q1[j][i])
    if len(f)==0:
        arr=np.array(f1)
        t=np.partition(arr, -10)[-10:]
        p=[]
        for j in range(0,len(f1)):
            if f1[j] in t:
                p.append(j)
        f2 = []
        for j in range(0, len(q)):
            f2.append(q[j][i])
        for j in p:
            f.append(f2[j])
    r=most_frequent_numbers(f)
    r=r[0]
    d1.append(r)

d1 = np.array(d1)
d = data_train.y[valid_idx].numpy()
k0 = 0
k1 = 1
a0 = 0
a1 = 0
for i in range(len(d)):
    if d[i] == 1:
        a1 = a1 + 1
    if d[i] == 0:
        a0 = a0 + 1
    if d[i] == 0 and d1[i] == d[i]:
        k0 = k0 + 1
    if d[i] == 1 and d1[i] == d[i]:
        k1 = k1 + 1
l = 0
for i in range(len(d)):
    if v[i] == d[i]:
        l = l + 1

print(k0 / a0)
print(k1 / a1)
print("acc:", l / len(d))
