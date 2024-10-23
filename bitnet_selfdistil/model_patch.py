from typing import List
import torch.nn as nn
from .bitlinear import BitLinearWithLoRA


def replace_linear(linear: nn.Linear, lora_rank: int) -> BitLinearWithLoRA:
    new_layer = BitLinearWithLoRA(
        weight=linear.weight,
        bias=linear.bias,
        lora_rank=lora_rank,
        standard_linear=False
    )
    return new_layer


def replace_linear_layers(model: nn.Module, blacklisted_modules: List[str], lora_rank: int) -> nn.Module:
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if name in blacklisted_modules:
            continue
        parent_name = '.'.join(name.split('.')[:-1])
        inside_parent_name = name.split('.')[-1]
        if isinstance(module, nn.Linear):
            new_module = replace_linear(module, lora_rank)
            setattr(modules[parent_name], inside_parent_name, new_module)
    return model


def selfdistil_train_mode(model: nn.Module, mode: bool) -> None:
    for module in model.modules():
        if not isinstance(module, BitLinearWithLoRA):
            module.train(False)
            for param in module.parameters(recurse=False):
                param.requires_grad = False
    for module in model.modules():
        if isinstance(module, BitLinearWithLoRA):
            module.train(mode)


def selfdistil_teacher_mode(model: nn.Module, mode: bool) -> None:
    for module in model.modules():
        if isinstance(module, BitLinearWithLoRA):
            module.standard_linear = mode


def selfdistil_init_lora(model: nn.Module, adapter_name: str) -> None:
    for module in model.modules():
        if isinstance(module, BitLinearWithLoRA):
            module.init_lora(adapter_name)


def selfdistil_set_trainable_adapters(model: nn.Module, trainable_adapters: List[str]) -> None:
    for module in model.modules():
        if isinstance(module, BitLinearWithLoRA):
            module.trainable_adapters = set(trainable_adapters)


class SelfDistilModelPatch:
    def __init__(self, model: nn.Module, lora_rank: int, blacklisted_modules: List[str]):
        self.model = replace_linear_layers(model, blacklisted_modules, lora_rank)
        self.lora_rank = lora_rank
        self.blacklisted_modules = blacklisted_modules

    def train(self, mode: bool = True) -> None:
        selfdistil_train_mode(self.model, mode)

    def teacher_mode(self, mode: bool = True) -> None:
        selfdistil_teacher_mode(self.model, mode)

    def init_lora(self, adapter_name: str) -> None:
        selfdistil_init_lora(self.model, adapter_name)

    def set_trainable_adapters(self, trainable_adapters: List[str]) -> None:
        selfdistil_set_trainable_adapters(self.model, trainable_adapters)
