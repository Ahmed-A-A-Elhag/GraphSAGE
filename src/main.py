import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


import torch_geometric
from torch_geometric.utils import to_dense_adj

from torch_scatter import scatter

import argparse

import torch_geometric
from torch_geometric.utils import to_dense_adj

from layer_Mean import GraphSAGE_Mean
from layer_MaxPooling import GraphSAGE_MaxPooling
from model import GraphSAGE

seed = 100
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, data, num_epochs, use_edge_index=False):
    if not use_edge_index:

        # Create the adjacency matrix
        adj = to_dense_adj(data.edge_index)[0]

    else:

        # Directly use edge_index, ignore this branch for now
        adj = data.edge_index
        
    model.to(device)
    data.to(device)

    # Set up the optimizer
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # A utility function to compute the accuracy
    def get_acc(outs, y, mask):
        return (outs[mask].argmax(dim=1) == y[mask]).sum().float() / mask.sum()

    best_acc_val = -1
    for epoch in range(num_epochs):

        # Zero grads -> forward pass -> compute loss -> backprop
        
        # train mode
        model.train()

        optimizer.zero_grad()
        outs = model(data.x.to(device), adj.to(device))

        # null_loss 

        loss = loss_fn(outs[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Compute accuracies, print only if this is the best result so far

        # evaluation mode
        model.eval()

        # data.x = the features of the dataset

        outs = model(data.x, adj)

        # validation accuracy 
        acc_val = get_acc(outs, data.y, data.val_mask)

        # test accuracy 
        acc_test = get_acc(outs, data.y, data.test_mask)

        # print the accuracy if it’s incresed
        if acc_val > best_acc_val:
            best_acc_val = acc_val
            print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Val: {acc_val:.3f} | Test: {acc_test:.3f}')

    print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Val: {acc_val:.3f} | Test: {acc_test:.3f}')
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--use-edge-index', type = bool ,default=True)
  parser.add_argument('--num-epochs', type = int, default=200)
  parser.add_argument('--data', type = str, default="Cora")
  
  parser.add_argument('--num-hid', type = int)

  

  args = parser.parse_args()

  if(args.data == 'Cora'):
        dataset = torch_geometric.datasets.Planetoid(root='/', name='Cora')
  if(args.data == 'CiteSeer'):
        dataset = torch_geometric.datasets.Planetoid(root='/', name='CiteSeer')

        
  model = GraphSAGE(nfeat = dataset.num_features, nhid = args.num_hid, nclass = dataset.num_classes)

  train(model, data = dataset[0] , num_epochs = args.num_epochs)
  
