import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant(w):
    scale = w.abs().mean()
    adjustment = 1e-4 + scale / 2
    w_quant = w / adjustment
    return torch.clip(input=torch.round(w_quant), min=-1, max=1) * scale


class BitnetDeltaWLinear(nn.Linear):
    def __init__(self, base_linear: nn.Linear):
        out_features, in_features = base_linear.weight.shape
        nn.Module.__init__(self)
        self.weight = base_linear.weight
        self.delta_weight = nn.Parameter(torch.zeros_like(self.weight))
        self.bias = base_linear.bias
        self.teacher = True
        self.in_features = in_features
        self.out_features = out_features

    def train(self, mode = True):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.delta_weight.requires_grad = mode

    def forward(self, input):
        if self.teacher:
            W = self.weight
        else:
            W = self.weight + self.delta_weight
            W = W + (weight_quant(W) - W).detach()
        return F.linear(input, W, self.bias)