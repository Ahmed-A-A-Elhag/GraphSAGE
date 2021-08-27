import torch
from torch_scatter import scatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphSAGE_Mean(torch.nn.Module):
    """
    GraphSAGE_Mean layer
    """

    def __init__(self, in_features, out_features, normalize = True, bias = False):  
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        self.bias = bias

        # linear transformation that apply to embedding for central node
        self.linear_l = torch.nn.Linear(self.in_features, self.out_features, bias = self.bias)
        
        #linear transformation that you apply to aggregated message from neighbors
        self.linear_r = torch.nn.Linear(self.in_features, self.out_features, bias = self.bias)




    def forward(self, fts, edge_index):

        out = None
        u, v = edge_index
        aggregate = scatter(fts[v].to(device), u.to(device), dim = 0, reduce='mean')
        
        fts = self.linear_l(fts)
        aggregate = self.linear_r(aggregate)

        out = fts + aggregate


        if self.normalize:
            out = out/torch.norm(out, dim=1).unsqueeze(-1)

        return out.to(device)

