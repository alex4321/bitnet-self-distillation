from transformers import Trainer


class MultiComponentLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)

        loss = outputs["loss"]
        log_dict = {
            "loss": loss.item(),
        }
        for key, value in outputs["losses"].items():
            log_dict[key] = value.item()
        self.log(log_dict)
        if return_outputs:
            return loss, outputs
        else:
            return loss
