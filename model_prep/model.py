from transformers import GPTNeoXForCausalLM

from .config import build_model_config


def build_model():
    config = build_model_config()
    model = GPTNeoXForCausalLM(config)
    return model