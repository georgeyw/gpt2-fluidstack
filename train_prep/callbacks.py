

from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy

class CustomDefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.evaluation_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if type(state.save_steps) == int: 
            if (
                args.save_strategy == IntervalStrategy.STEPS
                and state.save_steps > 0
                and state.global_step % state.save_steps == 0
            ):
                control.should_save = True
        else:
            if (
                args.save_strategy == IntervalStrategy.STEPS
                and state.global_step in state.save_steps
            ):
                control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control