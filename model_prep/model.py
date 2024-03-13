import torch
from transformers import GPTNeoXForCausalLM

from .config import build_model_config


def build_model(seed=5):
    torch.manual_seed(seed)
    config = build_model_config()
    model = GPTNeoXForCausalLM(config)
    return model

def load_init_model():
    model = GPTNeoXForCausalLM.from_pretrained(
            'georgeyw/gpt-2-small-init-seed-5', 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
    return model
