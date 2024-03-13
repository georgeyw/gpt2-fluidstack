from data_prep.data import build_dataset
from model_prep.model import build_model
from train_prep.trainer import CustomTrainer, custom_data_collator
from train import TRAIN_ARGS


if __name__ == "__main__":
    dataset = build_dataset()

    trainer = CustomTrainer(
        model_init=build_model,
        args=TRAIN_ARGS,
        train_dataset=dataset,
        data_collator=custom_data_collator,
    )

    trainer.push_to_hub('georgeyw/gpt-2-small-init-seed-5')
    