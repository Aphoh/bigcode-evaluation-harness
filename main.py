import argparse
import fnmatch
import json

import datasets
import transformers
from transformers import AutoTokenizer

from bigcode_eval.evaluator import Evaluator
from bigcode_eval.tasks import ALL_TASKS

def parse_args():
    parser = argparse.ArgumentParser(
        description="BigCode Evaluation Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common arguments
    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help=f"Evaluation tasks from {ALL_TASKS}, comma-separated with wildcard support",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--n_copies",
        type=int,
        default=1,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type to use for generation in tasks",
    )

    # Create subparsers for the two commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # write_inputs command
    write_parser = subparsers.add_parser(
        "write_inputs", 
        help="Generate input prompts to be passed to another tool"
    )
    write_parser.add_argument(
        "--output_file",
        type=str,
        default="generation_inputs.jsonl",
        help="Path to save the generation inputs as JSONL",
    )

    write_parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="Instruction tokens used for instruction-tuning benchmarks (comma-separated)",
    )
    write_parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'",
    )
    
    # eval command
    eval_parser = subparsers.add_parser(
        "eval", 
        help="Evaluate outputs from another tool"
    )
    eval_parser.add_argument(
        "--generations_file",
        type=str,
        required=True,
        help="Path to file with generated solutions to evaluate",
    )
    # eval_parser.add_argument(
    #     "--results_file",
    #     type=str,
    #     default="evaluation_results.json",
    #     help="Path to save the evaluation results",
    # )
    eval_parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    eval_parser.add_argument(
        "--postprocess",
        action="store_true",
        default=True,
        help="Postprocess model outputs before execution",
    )
    eval_parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't evaluate generations but benchmark groundtruth (useful for debugging)",
    )
    eval_parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Multithreaded Eval",
    )

    args = parser.parse_args()
    
    # Validate arguments
    if args.command is None:
        parser.error("Please specify a command: write_inputs or eval")
    
    if args.tasks is not None:
        # Validate tasks using pattern matching
        task_patterns = args.tasks.split(",")
        matched_tasks = pattern_match(task_patterns, ALL_TASKS)
        if not matched_tasks:
            parser.error(f"No tasks match the patterns: {args.tasks}")
    
    return args


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    # Determine which tasks to run
    if args.tasks is None:
        print("No tasks specified. Pick one of the following")
        print(", ".join(ALL_TASKS))
        return
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=getattr(args, "revision", None),
        trust_remote_code=args.trust_remote_code,
        token=getattr(args, "use_auth_token", None),
        truncation_side="left",
        padding_side="right",
    )
    
    # Configure tokenizer (same as before)
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    try:
        tokenizer.pad_token = tokenizer.eos_token
    except AttributeError:
        print("Not setting pad_token to eos_token")
        pass
        
    # Handle special case models
    WIZARD_LLAMA_MODELS = [
        "WizardLM/WizardCoder-Python-34B-V1.0",
        "WizardLM/WizardCoder-34B-V1.0",
        "WizardLM/WizardCoder-Python-13B-V1.0"
    ]
    if args.model in WIZARD_LLAMA_MODELS:
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = 1
        print("Changing bos_token to <s>")
    
    # Handle the different commands
    if args.command == "write_inputs":
        evaluator = Evaluator(tokenizer, args.prompt, args.load_data_path, False)
        print(f"Generating inputs for tasks: {', '.join(task_names)}")
        
        with open(args.output_file, "w") as f:
            for task in task_names:
                print(f"Processing task: {task}")
                # Get generation inputs but don't run evaluation
                prefix = args.prefix
                instruction_tokens = args.instruction_tokens
                evaluator.write_gen_inputs(task, f, args.limit, args.n_copies, prefix, instruction_tokens)
            
    elif args.command == "eval":
        evaluator = Evaluator(tokenizer, args.prompt, args.load_data_path, True)
        # Load generations from file
        results = evaluator.evaluate_generations(task_names, args)
        
        # Save all args to config
        results["config"] = vars(args)
        dumped = json.dumps(results, indent=2)
        print(dumped)
        
        results_file = args.generations_file[:args.generations_file.find("output")] + "eval.jsonl"
        with open(results_file, "w") as f:
            f.write(dumped)
        print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
