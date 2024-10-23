from typing import Type, Callable, List, Dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
import torch.nn as nn
import torch
from .model_patch import SelfDistilModelPatch
from .losses import SelfDistilLossesCalculator


@dataclass
class ReLoRAConfig:
    blacklisted_modules: List[str]
    lora_rank: int
    optimizer_type: Type[Optimizer]
    optimizer_kwargs: dict | None
    reset_steps: int
    chunk_warmup_steps: int
    lr_global: Callable[[int], float]

    def get_lambda_lr(self) -> Callable[[int], float]:
        def _lr(step: int) -> float:
            k = min((step % self.reset_steps) / self.chunk_warmup_steps, 1.0)
            lr = self.lr_global(step)
            return lr * k

        return _lr


@dataclass
class ReLoRAEvents:
    on_step_end: None | Callable[[int, Optimizer, Dict[str, torch.Tensor], torch.Tensor], None]
    on_chunk_end: None | Callable[[int, int], None]


class ReloraTrainer:
    def __init__(self, model: nn.Module,
                 relora_config: ReLoRAConfig,
                 events: ReLoRAEvents,
                 losses_calculator: SelfDistilLossesCalculator,
                 max_steps: int,
                 model_kwargs: dict | None):
        self.model = model
        self.patch = SelfDistilModelPatch(model, relora_config.lora_rank, relora_config.blacklisted_modules)
        self.losses_calculator = losses_calculator
        self.events = events
        self.relora_config = relora_config
        self.max_steps = max_steps
        self.model_kwargs = model_kwargs

    def _chunks(self, dataloader):
        batches = []
        for batch in dataloader:
            batches.append(batch)
            if len(batches) == self.relora_config.reset_steps:
                yield batches
                batches = []
        if len(batches) > 0:
            yield batches

    def _optimizer(self) -> Optimizer:
        if self.relora_config.optimizer_kwargs is None:
            optimizer_kwargs = {}
        else:
            optimizer_kwargs = self.relora_config.optimizer_kwargs
        return self.relora_config.optimizer_type(self.model.parameters(), **optimizer_kwargs)

    def _chunk_train(self, index: int, start_step: int, batches: List[dict]):
        if self.model_kwargs is not None:
            model_kwargs = self.model_kwargs
        else:
            model_kwargs = {}

        chunk_adapter_name = f'lora_{index}'
        self.patch.init_lora(adapter_name=chunk_adapter_name)
        self.patch.set_trainable_adapters([chunk_adapter_name])
        self.patch.train(True)

        optimizer = self._optimizer()
        scheduler = LambdaLR(optimizer, self.relora_config.get_lambda_lr())

        for i, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            self.patch.teacher_mode(True)
            with torch.no_grad():
                teacher_outputs = self.model(**batch, **model_kwargs)
            self.patch.teacher_mode(False)
            student_outputs = self.model(**batch, **model_kwargs)
            loss_components, loss = self.losses_calculator(teacher_outputs, student_outputs)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if self.events.on_step_end is not None:
                self.events.on_step_end(start_step + i, optimizer, loss_components, loss)

        optimizer.zero_grad(set_to_none=True)

    def train(self, dataloader_train):
        step = 0
        for i, batches in enumerate(self._chunks(dataloader_train)):
            self._chunk_train(i, step, batches)
            step += len(batches)
            if self.events.on_chunk_end is not None:
                self.events.on_chunk_end(i, step)
