import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3DecoderLayer


def patched_phi3RMS_norm_forward(module):
    def forward(hidden_states):
        assert (hidden_states.dtype == torch.bfloat16) or (hidden_states.dtype == torch.float32), \
            f"Only works with bfloat16 or float32 now, got {hidden_states.dtype}"
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
        return module.weight * hidden_states

    return forward


def any_requires_grad(*args) -> bool:
    for item in args:
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                return True
    return False


def patched_phi3_decoder_forward(module: Phi3DecoderLayer) -> callable:
    def forward(hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                **kwargs):
        if not any_requires_grad(hidden_states, attention_mask, position_ids, past_key_value):
            result = Phi3DecoderLayer.forward(
                module,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            result = checkpoint(
                lambda *args: Phi3DecoderLayer.forward(module, *args, **kwargs),
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                use_reentrant=True
            )
        return result
    return forward


def phi3_full_gradient_checkpoint_enable(model: Phi3ForCausalLM) -> Phi3ForCausalLM:
    model.enable_input_require_grads()
    for module in model.model.layers:
        module.forward = patched_phi3_decoder_forward(module)
        module.input_layernorm.forward = patched_phi3RMS_norm_forward(module.input_layernorm)
        module.post_attention_layernorm.forward = patched_phi3RMS_norm_forward(module.post_attention_layernorm)
    return model
