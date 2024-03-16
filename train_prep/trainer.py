import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# from huggingface_hub import upload_folder
from transformers import Trainer
from transformers.trainer_utils import HubStrategy, IntervalStrategy
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    is_peft_available,
)

from .upload import async_upload_to_s3

# transformers constants
PREFIX_CHECKPOINT_DIR = "checkpoint"
TRAINER_STATE_NAME = "trainer_state.json"
TRAINING_ARGS_NAME = "training_args.bin"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


# subclass Trainer for custom loss
# only doing this to avoid having to figure out how to add labels to the dataset
class CustomTrainer(Trainer):
    def __init__(self, push_aws_every=None, clear_threads_every=5_000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.push_hub_every = push_aws_every
        assert self.push_hub_every % self.args.save_steps == 0
        self.clear_threads_every = clear_threads_every
        self.threads = []

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = torch.stack(inputs)
        outputs = model(inputs)
        logits = outputs.logits
        loss = lm_cross_entropy_loss(logits, inputs)
        return (loss, outputs) if return_outputs else loss
    
    # overriding
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            if self.push_hub_every:
                if self.state.global_step > 0 and self.state.global_step % self.push_hub_every == 0:
                    last_global_save_step = self.state.global_step - self.push_hub_every
                    checkpoint_folders = [f"{PREFIX_CHECKPOINT_DIR}-{i}" for i in range(last_global_save_step + self.args.save_steps, self.state.global_step + 1, self.args.save_steps)]
                    self._push_from_checkpoints(checkpoint_folders)
            else:
                self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)


    # adding new function mimicking _push_from_checkpoint
    # probably not very efficient
    def _push_from_checkpoints(self, checkpoint_folders):
        # Only push from one node.
        if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
            return
        # If we haven't finished the last push, we don't do this one unless args.hub_always_push=True.
        if not self.args.hub_always_push and self.push_in_progress is not None and not self.push_in_progress.is_done():
            return

        output_dir = self.args.output_dir
        upload_dir = Path(output_dir + "-upload-" + str(self.state.global_step))
        os.makedirs(upload_dir, exist_ok=True)
        # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
        if is_peft_available():
            modeling_files.extend([ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME])
        for modeling_file in modeling_files:
            for checkpoint_folder in checkpoint_folders:
                os.makedirs(os.path.join(upload_dir, checkpoint_folder), exist_ok=True)
                if os.path.isfile(os.path.join(output_dir, checkpoint_folder, modeling_file)):
                    shutil.copy(os.path.join(output_dir, checkpoint_folder, modeling_file), os.path.join(upload_dir, checkpoint_folder, modeling_file))
        # Saving the tokenizer is fast and we don't know how many files it may have spawned, so we resave it to be sure.
        if self.tokenizer is not None:
            for checkpoint_folder in checkpoint_folders:
                self.tokenizer.save_pretrained(os.path.join(upload_dir, checkpoint_folder))
        # Same for the training arguments
        for checkpoint_folder in checkpoint_folders:
            torch.save(self.args, os.path.join(upload_dir, checkpoint_folder, TRAINING_ARGS_NAME))

        if self.args.save_strategy == IntervalStrategy.STEPS:
            commit_message = f"Training in progress, step {self.state.global_step}"
        else:
            commit_message = f"Training in progress, epoch {int(self.state.epoch)}"

        # model_push_job = upload_folder(
        #     repo_id=self.hub_model_id,
        #     folder_path=output_dir,
        #     commit_message=commit_message,
        #     token=self.args.hub_token,
        #     run_as_future=True,
        #     ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
        # )

        # push_jobs = [model_push_job]
        push_jobs = []

        if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
            # checkpoint_push = upload_folder(
            #     repo_id=self.hub_model_id,
            #     folder_path=upload_dir,
            #     commit_message=commit_message + ", checkpoint",
            #     token=self.args.hub_token,
            #     run_as_future=True,
            # )
            # push_jobs.append(checkpoint_push)
            thread = async_upload_to_s3(upload_dir)
            self.threads.append(thread)

        # close threads
        if self.state.global_step > 0 and self.state.global_step % self.clear_threads_every == 0:
            if len(self.threads) > 0:
                last_thread = self.threads[-1]
                for thread in self.threads[:-1]:
                    thread.join()
                self.threads = [last_thread]

        if self.push_in_progress is None or self.push_in_progress.is_done():
            self.push_in_progress = PushInProgress(push_jobs)
        else:
            self.push_in_progress.jobs.extend(push_jobs)
    

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
    

def custom_data_collator(features):
    # this is a dummy data collator that just returns the input
    return features
