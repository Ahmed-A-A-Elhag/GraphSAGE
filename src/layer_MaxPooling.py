import torch
from torch_scatter import scatter

class GraphSAGE_MaxPooling(torch.nn.Module):
    """
    GraphSAGE_MaxPooling layer
    """

    def __init__(self, in_features, out_features, normalize = True, bias = False):  
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        self.bias = bias

        # linear transformation that apply after concatenation 
        self.linear_l = torch.nn.Linear(2*self.in_features, self.out_features, bias = self.bias)
        
        #linear transformation that you apply to neighbors features before max pooling
        self.linear_r = torch.nn.Linear(self.in_features, self.in_features, bias = self.bias)

        # non-linearity before pooling
        self.relu = torch.nn.ReLU()


    def forward(self, fts, edge_index):

        
        out = None
        u, v = edge_index
        
        aggregate = scatter(self.relu(fts[v].to(device)), u.to(device), dim=0, reduce="max")
        

        # aggregate = self.linear_r(fts.to(device))
        # aggregate = self.relu(aggregate)
        # aggregate = scatter(aggregate[v], u.to(device), dim = 0, reduce='max')


        out = torch.cat([fts, aggregate], dim= 1)

   
        out = self.linear_l(out)

        if self.normalize:
            out = out/torch.norm(out, dim=1).unsqueeze(-1)

        return out.to(device)

