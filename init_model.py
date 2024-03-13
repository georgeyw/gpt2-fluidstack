from data_prep.data import build_dataset
from model_prep.model import build_model
from train_prep.trainer import custom_data_collator
from train import TRAIN_ARGS

from transformers import Trainer


if __name__ == "__main__":
    dataset = build_dataset()

    trainer = Trainer(
        model_init=build_model,
        args=TRAIN_ARGS,
        train_dataset=dataset,
        data_collator=custom_data_collator
    )

    trainer._save_checkpoint(trainer.model, trial=None)
