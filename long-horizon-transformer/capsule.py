# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

class Squash(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, s: torch.Tensor):
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))
        
class Router(nn.Module):
    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):
        super().__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()
        
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d), requires_grad=True)
        
def forward(self, u: torch.Tensor):
    u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)
    b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)
    v = None
    for i in range(self.iterations):
        c = self.softmax(b)
        s = torch.einsum('bij,bijm->bjm', c, u_hat)
        v = self.squash(s)
        a = torch.einsum('bjm,bijm->bij', v, u_hat)
        b = b + a
    return v

class MarginLoss(nn.Module):
    def __init__(self, *, n_labels: int, lambda_: float = 0.5, m_positive: float = 0.9, m_negative: float = 0.1):
        super().__init__()
        self.m_negative = m_negative
        self.m_positive = m_positive
        self.lambda_ = lambda_
        self.n_labels = n_labels
    
    def forward(self, v: torch.Tensor, labels: torch.Tensor):
        v_norm = torch.sqrt((v ** 2).sum(dim=-1))
        labels = torch.eye(self.n_labels, device=labels.device)[labels]
        loss = labels * F.relu(self.m_positive - v_norm) + \
            self.lambda_ * (1.0 - labels) * F.relu(v_norm - self.m_negative)
        return loss.sum(dim=-1).mean()
    


        































        
        
        
        
        