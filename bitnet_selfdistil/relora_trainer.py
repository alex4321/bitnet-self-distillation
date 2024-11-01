from typing import Type, Callable, List, Dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass
import torch.nn as nn
import torch
from enum import Enum
import os
import logging
from .model_patch import SelfDistilModelPatch
from .losses import SelfDistilLossesCalculator


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StopCondition(Enum):
    CONTINUE = 0
    STOP = 1


@dataclass
class ReLoRAConfig:
    blacklisted_modules: List[str]
    lora_rank: int
    optimizer_type: Type[Optimizer]
    optimizer_kwargs: dict | None
    reset_steps: int
    chunk_warmup_steps: int
    lr_global: Callable[[int], float]


@dataclass
class ReLoRAEvents:
    on_step_end: None | Callable[["ReloraTrainer", int, Optimizer, Dict[str, torch.Tensor], torch.Tensor], None]
    on_chunk_end: None | Callable[["ReloraTrainer", int, int], StopCondition]


class ReloraTrainer:
    def __init__(self, model: nn.Module,
                 relora_config: ReLoRAConfig,
                 events: ReLoRAEvents,
                 losses_calculator: SelfDistilLossesCalculator,
                 model_kwargs: dict | None,
                 checkpoint_directory: str | None):
        self.model = model
        self.patch = SelfDistilModelPatch(model, relora_config.lora_rank, relora_config.blacklisted_modules)
        self.losses_calculator = losses_calculator
        self.events = events
        self.relora_config = relora_config
        self.model_kwargs = model_kwargs
        self.checkpoint_directory = checkpoint_directory

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
        optimizer = self.relora_config.optimizer_type(self.model.parameters(), **optimizer_kwargs)
        logger.info("Optimizer initialized with kwargs=%s", optimizer_kwargs)
        return optimizer
    
    def _move_batch(self, batch):
        device = self.model.device
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device=device)
            else:
                result[key] = value
        logger.debug("Batch moved to device %s", device)
        return result
    
    def _chunk_adapter_name(self, index: int) -> str:
        return f"lora_{index}"

    def _chunk_train(self, index: int, start_step: int, batches: List[dict]) -> None:
        logger.info("Starting training on chunk %d with start_step=%d", index, start_step)
        if self.model_kwargs is not None:
            model_kwargs = self.model_kwargs
        else:
            model_kwargs = {}

        chunk_adapter_name = self._chunk_adapter_name(index)
        self.patch.init_lora(adapter_name=chunk_adapter_name)
        self.patch.set_trainable_adapters([chunk_adapter_name])
        self.patch.train(True)

        optimizer = self._optimizer()

        def _lambda_lr(step: int) -> float:
            lr_global = self.relora_config.lr_global(step + start_step)
            chunk_warmup_k = min(1.0, step / self.relora_config.chunk_warmup_steps)
            return lr_global * chunk_warmup_k

        scheduler = LambdaLR(optimizer, _lambda_lr)

        for i, batch in enumerate(batches):
            batch = self._move_batch(batch)
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

            logger.debug("Step %d, loss=%f", start_step + i, loss.item())

            if self.events.on_step_end is not None:
                self.events.on_step_end(self, start_step + i, optimizer, loss_components, loss)

        optimizer.zero_grad(set_to_none=True)
        logger.info("Completed training on chunk %d", index)

    def _checkpoint_dict(self, chunk_index: int, step: int) -> dict:
        adapter_name = self._chunk_adapter_name(chunk_index)
        adapter_weights = {
            key: value
            for key, value in self.model.state_dict().items()
            if adapter_name in key and (("lora_A" in key) or ("lora_B" in key))
        }
        return {
            "chunk_index": chunk_index,
            "adapter_name": adapter_name,
            "adapter_weights": adapter_weights,
            "step": step,
        }
    
    def _checkpoint_file(self, chunk_index: int, checkpoint_directory: str) -> str:
        return os.path.join(checkpoint_directory, f"chunk-{chunk_index}.ckpt")
    
    def _save_checkpoint(self, chunk_index: int, step: int, checkpoint_directory: str) -> None:
        os.makedirs(checkpoint_directory, exist_ok=True)
        data = self._checkpoint_dict(chunk_index, step)
        torch.save(data, self._checkpoint_file(chunk_index, checkpoint_directory))
        logger.info("Saved checkpoint for chunk %d at step %d", chunk_index, step)

    def _load_checkpoint(self, chunk_index: int, checkpoint_dir: str):
        fname = self._checkpoint_file(chunk_index, checkpoint_dir)
        checkpoint_data = torch.load(fname)
        adapter_name = checkpoint_data["adapter_name"]
        self.patch.init_lora(adapter_name)
        self.model.load_state_dict(dict(
            self.model.state_dict(),
            **checkpoint_data["adapter_weights"]
        ))
        step = checkpoint_data["step"]
        logger.info("Loaded checkpoint for chunk %d, resuming from step %d", chunk_index, step)
        return step

    def train(self, dataloader_train, continue_from_checkpoint):
        if continue_from_checkpoint:
            assert self.checkpoint_directory is not None, \
                "To use continue_from_checkpoint=True you need to set checkpoint_directory"
        logger.info("Starting training with continue_from_checkpoint=%s", continue_from_checkpoint)
        step = 0
        for i, batches in enumerate(self._chunks(dataloader_train)):
            if continue_from_checkpoint and os.path.exists(self._checkpoint_file(i, self.checkpoint_directory)):
                step = self._load_checkpoint(i, self.checkpoint_directory)
            else:
                self._chunk_train(i, step, batches)
                step += len(batches)
                if self.checkpoint_directory is not None:
                    self._save_checkpoint(i, step, self.checkpoint_directory)
                if self.events.on_chunk_end is not None:
                    stop_condition = self.events.on_chunk_end(self, i, step)
                    if stop_condition == StopCondition.STOP:
                        logger.info("Training stopped at chunk %d, step %d by stop condition", i, step)
                        break
        logger.info("Training completed")
    
    def evaluate(self, batches: List[dict]) -> Dict[str, float]:
        if self.model_kwargs is not None:
            model_kwargs = self.model_kwargs
        else:
            model_kwargs = {}
        # Set model to evaluation mode
        self.patch.train(False)  # Set to eval mode for the patch
        eval_loss_components = []
        with torch.no_grad():  # Disable gradient calculation for evaluation
            for batch in batches:
                batch = self._move_batch(batch)
                # Run teacher model inference
                self.patch.teacher_mode(True)
                teacher_outputs = self.model(**batch, **model_kwargs)
                # Run student model inference
                self.patch.teacher_mode(False)
                student_outputs = self.model(**batch, **model_kwargs)
                # Calculate loss components and aggregate
                loss_components, _ = self.losses_calculator(teacher_outputs, student_outputs)
                eval_loss_components.append(loss_components)
            # Aggregate losses for all batches in this chunk
            avg_loss_components = {
                key: torch.stack([lc[key] for lc in eval_loss_components]).mean().item()
                for key in eval_loss_components[0]
            }
        logger.info("Evaluation completed with average loss components: %s", avg_loss_components)
        return avg_loss_components