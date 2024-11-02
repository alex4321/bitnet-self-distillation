import torch
import torch.nn as nn
from .model_patch import teacher_mode, train_mode


class TrainerModel(nn.Module):
    def __init__(self, model, losses_calculator):
        super(TrainerModel, self).__init__()
        self.model = model
        self.losses_calculator = losses_calculator

    def train(self, mode=True):
        train_mode(self, mode)

    def eval(self):
        train_mode(self, False)

    def forward(self, input_ids, attention_mask, labels):
        teacher_mode(self.model, True)
        with torch.no_grad():
            teacher_output = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)
        teacher_mode(self.model, False)
        student_output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    output_hidden_states=True)
        losses, loss = self.losses_calculator(teacher_output, student_output)
        return {
            "loss": loss,
            "losses": losses,
        }
