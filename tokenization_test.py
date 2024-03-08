import datasets
import torch

from transformers import GPT2TokenizerFast
from lang.transformer_lens import tokenize_and_concatenate

from constants import REPO_ID, CONTEXT_LENGTH, BATCH_SIZE, PYTHIA_DS_COL

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

dataset = datasets.load_dataset(REPO_ID, split='train')
print('loaded dataset')

tokens_dataset = tokenize_and_concatenate(dataset,
                                          tokenizer,
                                          streaming=False,
                                          max_length=CONTEXT_LENGTH,
                                          column_name=PYTHIA_DS_COL,
                                          add_bos_token=True,
                                          num_proc=12)
data_loader = torch.utils.data.DataLoader(tokens_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)