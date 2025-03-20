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
    def __init__(self, tokenizer, prompt, load_data_path, allow_code_execution):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.load_data_path = load_data_path
        self.allow_code_execution = allow_code_execution

    
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


    def evaluate_generations(self, task_names, args):
        generations_path = args.generations_file
        print(f"Loading generations from {generations_path}")
        with open(generations_path, "r") as f:
            generations_data = [json.loads(line) for line in f]
        
        # Group generations by task
        task_generations = {}
        for entry in generations_data:
            task = entry["task_name"]
            if task not in task_generations:
                task_generations[task] = []
            task_generations[task].append(entry)
        
        # Run evaluation for each task
        results = {}
        for task_name in task_names:
            if task_name not in task_generations:
                print(f"Warning: No generations found for task {task}")
                continue
                
            print(f"Evaluating task: {task_name}")
            generations = task_generations[task_name]

            task = tasks.get_task(task_name, args)
            dataset = task.get_dataset()
            n_samples = min(args.limit, len(dataset)) if args.limit else len(dataset)
            references = [task.get_reference(dataset[i]) for i in range(n_samples)]
            
            gen_by_sample = defaultdict(list)
            for generation in generations:
                assert generation["task_name"] == task_name
                sample = generation["sample"]
                output = generation["output"]
                #find the earliest stop_word
                fim_middle = '<fim_middle>' if 'starcoder' in self.tokenizer.name_or_path else '<|fim_middle|>' #qwen
                completion_start = output.find(fim_middle) + len(fim_middle)
                output = output[completion_start:]
                completion_end = len(output)
                for stop_word in task.stop_words:
                    if stop_word in output:
                        completion_end = min(completion_end, output.find(stop_word))
                output = output[:completion_end]
                gen_by_sample[sample].append(output)

            predictions = []
            for i in range(n_samples):
                predictions.append(gen_by_sample.get(i, []))

            # make sure tokenizer plays nice with multiprocessing
            if task.requires_execution and not self.allow_code_execution:
                raise ValueError(_WARNING)
            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            task_result = task.process_results(predictions, references)
            results[task_name] = task_result
        return results
