import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

def init_W(embed_dim, in_dim):
    k = np.sqrt(1 / in_dim)
    return 2*k*torch.rand(embed_dim, in_dim) - k

def init_b(embed_dim, in_dim):
    k = np.sqrt(1 / in_dim)
    return 2*k*torch.rand(embed_dim,) - k


class GNM_Module(nn.Module):
    def __init__(self, in_dim, embed_dim, activation):
        super().__init__()

        self.beta = nn.Parameter(torch.rand(1), requires_grad=True)
        self.W = nn.Parameter(init_W(embed_dim, in_dim), requires_grad=True)
        self.b = nn.Parameter(init_b(embed_dim, in_dim), requires_grad=True)
        self.act = activation()

    def forward(self, x):

        z = F.linear(x, weight = self.W, bias=self.b)
        z = self.act(z * F.softplus(self.beta))
        z = F.linear(z, weight=self.W.T)
        
        #J = F.softplus(self.beta) * self.W.T @ J_sigma @ self.W
        return z
    
class mGradNet_M(nn.Module):
    def __init__(self, num_modules, in_dim, embed_dim, activation):
        super().__init__()

        self.num_modules = num_modules
        self.mmgn_modules = nn.ModuleList([GNM_Module(in_dim, embed_dim, activation) for i in range(num_modules)])
        self.alpha = nn.Parameter(torch.rand(num_modules,), requires_grad=True)
        self.bias = nn.Parameter(init_b(in_dim, embed_dim), requires_grad=True)

    def forward(self, x):

        z = 0
        J = 0
        for i in range(self.num_modules):
            out = self.mmgn_modules[i](x)
            J_i = torch.func.vmap(torch.func.jacfwd(self.mmgn_modules[i]))(x)
            J += J_i * F.softplus(self.alpha[i])
            z += F.softplus(self.alpha[i]) * out
        logdet = torch.logdet(J)
        z += self.bias
        return z, logdet


class GradNet_M(nn.Module):
    def __init__(self, num_modules, in_dim, embed_dim, activation):
        super().__init__()

        self.num_modules = num_modules
        self.mmgn_modules = nn.ModuleList([GNM_Module(in_dim, embed_dim, activation) for i in range(num_modules)])
        self.alpha = nn.Parameter(torch.randn(num_modules,), requires_grad=True)
        self.bias = nn.Parameter(init_b(in_dim, embed_dim), requires_grad=True)

    def forward(self, x):
        z = 0
        J = 0
        for i in range(self.num_modules):
            out = self.mmgn_modules[i](x)
            J_i = torch.func.vmap(torch.func.jacfwd(self.mmgn_modules[i]))(x)
            J += J_i * self.alpha[i]
            z += self.alpha[i] * out
        logdet = torch.linalg.slogdet(J).logabsdet
        z += self.bias
        return z, logdet
