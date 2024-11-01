import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
from .losses import SelfDistilLosses, SelfDistilLossesCalculator


def _lm_losses(teacher: CausalLMOutput, student: CausalLMOutput, max_full_losses_length: int) -> SelfDistilLosses:
    def _loss_lm():
        loss_lm = student.loss
        return {
            "loss_lm": loss_lm,
        }

    def _additional_losses():
        kldiv_loss = F.kl_div(
            F.log_softmax(student.logits, dim=-1).view((-1, student.logits.shape[-1])),
            F.log_softmax(teacher.logits, dim=-1).view((-1, teacher.logits.shape[-1])),
            log_target=True,
            reduction="batchmean",
        )
        hidden_state_loss = 0
        for teacher_hidden_state, student_hidden_state in zip(teacher.hidden_states, student.hidden_states):
            teacher_hidden_state = teacher_hidden_state.view((-1, teacher_hidden_state.shape[-1]))
            student_hidden_state = student_hidden_state.view((-1, student_hidden_state.shape[-1]))
            hidden_state_loss_item = F.cosine_embedding_loss(
                student_hidden_state,
                teacher_hidden_state,
                torch.ones((teacher_hidden_state.shape[0],), device=teacher_hidden_state.device, dtype=torch.long),
            )
            hidden_state_loss += hidden_state_loss_item
        return {
            "kldiv_loss": kldiv_loss,
            "hidden_state_loss": hidden_state_loss,
        }

    def _all_losses() -> SelfDistilLosses:
        losses = _loss_lm()
        if student.logits.shape[0] * student.logits.shape[1] <= max_full_losses_length:
            losses = dict(losses, **_additional_losses())
        loss = 0
        for _, loss_value in losses.items():
            loss += loss_value
        losses["loss"] = loss
        return losses, loss

    return _all_losses()


def lm_losses_calculator(max_full_losses_length: int) -> SelfDistilLossesCalculator:
    def _lm_losses_inner(teacher: CausalLMOutput, student: CausalLMOutput) -> SelfDistilLosses:
        return _lm_losses(teacher, student, max_full_losses_length)

    return _lm_losses_inner
