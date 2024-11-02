import torch
import torch.nn as nn
from .bitnet import BitnetDeltaWLinear


def patch_model(model):
    modules = dict(model.named_modules())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (name not in ['lm_head']):
            parent_name = ".".join(name.split(".")[:-1])
            parent_module = modules[parent_name]
            setattr(parent_module, name.split(".")[-1], BitnetDeltaWLinear(module))
    return model


def teacher_mode(model, mode):
    for module in model.modules():
        if isinstance(module, BitnetDeltaWLinear):
            module.teacher = mode


def train_mode(model, mode):
    for module in model.modules():
        if not module is model:
            module.train(mode)
            for param in module.parameters():
                param.requires_grad = False
    for module in model.modules():
        if isinstance(module, BitnetDeltaWLinear):
            module.train(mode)