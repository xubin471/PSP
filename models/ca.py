
import torch
import torch.nn as nn
import torch.nn.functional as F

class CA(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.alpha = nn.Parameter(torch.zeros(1),requires_grad=True)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.query.out_features ** 0.5)
        attn = F.softmax(scores, dim=-1)
        new_proto = torch.matmul(attn, v)
        x1  = x1 + new_proto*self.alpha
        return x1