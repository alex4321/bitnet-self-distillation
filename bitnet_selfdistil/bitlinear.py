import torch.nn as nn
import torch.nn.functional as F
import torch
import math


def weight_quant(w):
    scale = w.abs().mean()
    adjustment = 1e-4 + scale / 2
    w_quant = w / adjustment
    return torch.clip(input=torch.round(w_quant), min=-1, max=1) * scale


class BitLinearWithLoRA(nn.Linear):
    """
    Replace original Linear layer using its weight + bias.
    """
    def __init__(self, weight: nn.Parameter, bias: nn.Parameter, lora_rank: int, standard_linear: bool = False):
        nn.Module.__init__(self)
        self.weight = weight
        self.bias = bias
        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.standard_linear = standard_linear
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.trainable_adapters = set()

    def init_lora(self, adapter_name: str) -> None:
        if adapter_name not in self.lora_A:
            self.lora_A[adapter_name] = nn.Linear(in_features=self.in_features,
                                                  out_features=self.lora_rank,
                                                  bias=False).to(
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            self.lora_B[adapter_name] = nn.Linear(in_features=self.lora_rank,
                                                  out_features=self.out_features,
                                                  bias=False).to(
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(1))
        nn.init.zeros_(self.lora_B[adapter_name].weight)

    def train(self, mode: bool = True) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        for adapter_name in self.lora_A.keys():
            if adapter_name in self.trainable_adapters:
                adapter_trainable = mode
            else:
                adapter_trainable = False
            self.lora_A[adapter_name].train(adapter_trainable)
            self.lora_B[adapter_name].train(adapter_trainable)
            for param in list(self.lora_A[adapter_name].parameters()) + list(self.lora_B[adapter_name].parameters()):
                param.requires_grad = adapter_trainable

    def get_adapter_weight_delta(self, adapter_name: str) -> torch.Tensor:
        # out_features x lora_rank
        lora_B = self.lora_B[adapter_name].weight
        # lora_rank x in_features
        lora_A = self.lora_A[adapter_name].weight
        adapter_diff = lora_B @ lora_A
        return adapter_diff

    def get_weight_delta(self) -> torch.Tensor:
        diff = None
        for adapter_name in self.lora_A.keys():
            adapter_diff = self.get_adapter_weight_delta(adapter_name)
            if diff is None:
                diff = adapter_diff
            else:
                diff += adapter_diff
        return diff

    def get_bitnet_weight(self) -> torch.Tensor:
        with torch.no_grad():
            W = self.weight + self.get_weight_delta()
            return weight_quant(W)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.standard_linear:
            W = self.weight
        else:
            W = self.weight
            W = W + self.get_weight_delta()
            W = W + (weight_quant(W) - W).detach()
        return F.linear(input, W, self.bias)
