from typing import Optional
from bigcode_eval.utils import FormattedDataset

def get_generation_inputs(
        task,
        task_name: str,
        dataset,
        tokenizer,
        n_samples: int,
        n_copies: int,
        prefix: Optional[str],
        instruction_tokens: Optional[str],
):
    extra_keys = {}
    # The input_length / start_length set to 0 for now will be adjusted later
    # Check if the task has a custom check_fn method for the stopping criteria
    if task.stop_words:
        extra_keys["stop_words"] = task.stop_words
        if tokenizer.eos_token:
            extra_keys["stop_words"].append(tokenizer.eos_token)
    if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
        extra_keys["max_length_multiplier"] = task.max_length_multiplier

    assert not hasattr(task, "check_fn"), "Custom check_fn is not supported"
        
    if instruction_tokens:
        instruction_tokens = instruction_tokens.split(",")
        if len(instruction_tokens) != 3:
            raise ValueError(
                "Instruction tokens should contain exactly 3 tokens separated by a comma. If a token is empty, represent it as ''"
            )
        for token in instruction_tokens:
            if token.strip() != "":
                task.stop_words.append(token)
    else:
        instruction_tokens = None

    print(f"number of problems for this task is {n_samples}")
    formatted_ds = FormattedDataset(
        task,
        task_name,
        dataset,
        tokenizer,
        n_samples=n_samples,
        n_copies=n_copies,
        prefix=prefix,
        instruction_tokens=instruction_tokens,
    )

    for elem in formatted_ds:
        yield elem | extra_keys


