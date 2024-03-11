from transformers import GPTNeoXConfig

# EleutherAI claims that Pythia and GPT-NeoX-20B use the same tokenizer
# However, Pythia's tokenizer has 50304 tokens while GPT-NeoX-20B's tokenizer has 50432 tokens
# Using Pythia's number bc we're using Pythia's preshuffled, pretokenized data
VOCAB_SIZE = 50304


def build_model_config():
    config = GPTNeoXConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        rotary_pct=0.25, # default
        rotary_emb_base=10_000, # default
        attention_dropout=0.0, # default
        hidden_dropout=0.0, # default
        classifier_dropout=0.1, # default
        max_position_embeddings=1024,
        initializer_range=0.02, # default (of 20B, not the default in the library though for some reason?)
        layer_norm_epsilon=1e-5, # default
        use_cache=True, # default
        bos_token_id=0, # default
        eos_token_id=2, # default
        tie_word_embeddings=False, # default
        use_parallel_residual=True, # default
        rope_scaling=None, # default
        attention_bias=True, # default
    )
    return config