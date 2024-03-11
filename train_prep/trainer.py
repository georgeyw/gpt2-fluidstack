import torch.nn.functional as F

from transformers import Trainer


# subclass Trainer for custom loss
# only doing this to avoid having to figure out how to add labels to the dataset
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        logits = outputs.logits
        loss = lm_cross_entropy_loss(logits, inputs)
        return (loss, outputs) if return_outputs else loss
    

# stolen from transformer_lens
def lm_cross_entropy_loss(
    logits,
    tokens,
    per_token: bool = False,
):
    """Cross entropy loss for the language model, gives the loss for predicting the NEXT token.

    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        per_token (bool, optional): Whether to return the log probs predicted for the correct token, or the loss (ie mean of the predicted log probs). Note that the returned array has shape [batch, seq-1] as we cannot predict the first token (alternately, we ignore the final logit). Defaults to False.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(
        dim=-1, index=tokens[..., 1:, None]
    )[..., 0]
    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.mean()
    