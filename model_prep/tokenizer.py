from transformers import GPTNeoXTokenizerFast

def build_tokenizer():
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/pythia-14m")
    return tokenizer