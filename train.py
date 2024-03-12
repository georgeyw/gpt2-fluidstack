import os

from transformers import TrainingArguments

from data_prep.data import build_dataset
from data_prep.constants import BATCH_SIZE, GRAD_ACCUM_STEPS
from model_prep.model import build_model
from train_prep.trainer import CustomTrainer, custom_data_collator

HUB_TOKEN = os.environ.get("HUB_TOKEN")
os.environ["WANDB_PROJECT"] = "gw-hf-trainer-test"

TRAIN_ARGS = TrainingArguments(
    output_dir='./checkpoints',
    overwrite_output_dir=True,
    do_train=True,
    torch_compile=True,
    deepspeed="./train_prep/ds_config.json",
    bf16=True,
    # tf32=True, # maybe change / test? compare with fp16 -- I think this is getting overwritten by deepspeed? with AMP
    # fp16=True,
    # fp16_opt_level="O1",
    # fp16_backend="apex",
    seed=5,
    full_determinism=True,
    # data
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=1,
    dataloader_pin_memory=True,
    dataloader_num_workers=0, # maybe change / test?
    # optim -- I think this is getting overwritten by deepspeed?
    optim="adamw_torch",
    learning_rate=6e-4,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    # checkpointing
    save_steps=100,
    push_to_hub=True,
    hub_model_id="georgeyw/gpt-2-small",
    hub_strategy="every_save",
    hub_always_push=True,
    hub_token=HUB_TOKEN,
    # wandb -- is this necessary with deepspeed?
    report_to="wandb",
    logging_steps=1,
    run_name="gw-hf-trainer-test-run-name",
)

if __name__ == "__main__":
    dataset = build_dataset()

    trainer = CustomTrainer(
        model_init=build_model,
        args=TRAIN_ARGS,
        train_dataset=dataset,
        data_collator=custom_data_collator,
    )

    trainer.train()
