
import torch 

from layer_Mean import GraphSAGE_Mean
from layer_MaxPooling import GraphSAGE_MaxPooling

class GraphSAGE(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, aggregator = 'Mean'):
        super().__init__()

        if(aggregator == 'Mean'):
            self.gc1 = GraphSAGE_Mean(nfeat, nhid)
            self.gc2 = GraphSAGE_Mean(nhid, nclass)
            

        elif(aggregator == 'MaxPooling'):
            self.gc1 = GraphSAGE_MaxPooling(nfeat, nhid)
            self.gc2 = GraphSAGE_MaxPooling(nhid, nclass)

        self.relu = torch.nn.ReLU()

    def forward(self, fts, adj):
        fts = self.relu(self.gc1(fts, adj))
        fts = self.gc2(fts, adj)
        return fts
