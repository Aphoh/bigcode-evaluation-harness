from collections import defaultdict
import inspect
import json
import os
from pathlib import Path
from typing import TextIO

from bigcode_eval import tasks
from bigcode_eval.generation import get_generation_inputs

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""

class Evaluator:
    def __init__(self, tokenizer, prompt, load_data_path):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.load_data_path = load_data_path

    
    def write_gen_inputs(self, task_name: str, handle: TextIO, limit: int, n_copies: int, prefix: str, instruction_tokens: str):
        task = tasks.get_task(task_name, self.prompt, self.load_data_path)
        dataset = task.get_dataset()
        n_samples = min(limit, len(dataset)) if limit else len(dataset)
        gen_inputs = get_generation_inputs(
            task,
            task_name,
            dataset,
            self.tokenizer,
            n_samples,
            n_copies=n_copies,
            prefix=prefix,
            instruction_tokens=instruction_tokens,
        )
        # Write the generations to a jsonl
        for g in gen_inputs:
            handle.write(json.dumps(g) + "\n")
        print(f"Wrote task {task_name}")


    def evaluate_generations(self, task_name, generations_path: Path, allow_code_execution: bool):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        n_samples = min(self.args.limit, len(dataset)) if self.args.limit else len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(n_samples)]

        # TODO: what is this doing? checking a ground truth maybe?
        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(n_samples)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        gen_by_sample = defaultdict(list)
        with open(generations_path, "r") as f:
            for line in f:
                if line.strip():
                    elem = json.loads(line)
                    assert elem["task_name"] == task_name
                    sample = elem["sample"]
                    gen_by_sample[sample].append(elem["output"])

        generations = []
        for i in range(n_samples):
            generations.append(gen_by_sample.get(i, []))

        if task.requires_execution and not allow_code_execution:
            raise ValueError(_WARNING)
        # make sure tokenizer plays nice with multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print("Evaluating generations...")
        results = task.process_results(generations, references)
        return results