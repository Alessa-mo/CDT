import argparse
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import vllm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_data(file_path: Path) -> List[Dict[str, Any]]:
    logging.info(f"Loading data from {file_path}...")
    if file_path.suffix == ".jsonl":
        with file_path.open('r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    elif file_path.suffix == ".json":
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Please provide a .json or .jsonl file.")

def generate_prompt(
    prompt_template: Dict[str, Any],
    skill_definitions: Dict[str, Any],
    instruction: str,
    tag_type: str
) -> str:
    sys_prompt = prompt_template.get('system_prompt', '')
    template = prompt_template.get('prompt_template', '{question}')
    defaults = prompt_template.get('defaults', {})
 
    skill_names = list(skill_definitions.keys())
    random.shuffle(skill_names)

    # Format the skill descriptions
    skills_text = ""
    if tag_type == "domain":
        for i, name in enumerate(skill_names, 1):
            subdomains = skill_definitions[name]
            skills_text += f"\n{name}: {', '.join(subdomains)}"
    else:
        for name in skill_names:
            definition = skill_definitions[name]
            skills_text += f"\n{name}: {definition}"
            
    prompt = template.format(question=instruction, skill=skills_text, **defaults)
    return f"{sys_prompt} {prompt}"

class AnnotationDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]

def prepare_prompts(
    data_list: List[Dict[str, Any]],
    prompt_templates: List[Dict[str, Any]],
    prompt_id: int,
    skill_definitions: Dict[str, Any],
    tag_type: str
) -> List[str]:
    logging.info("Preparing prompts...")
    
    # Find the corresponding template based on prompt_id
    try:
        prompt_template = next(item for item in prompt_templates if item["prompt_id"] == prompt_id)
    except StopIteration:
        raise ValueError(f"Template with ID {prompt_id} not found in the prompt file.")

    prompts = []
    example_printed = False
    for item in tqdm(data_list, desc="Generating Prompts"):

        if 'messages' in item and isinstance(item['messages'], list) and len(item['messages']) > 0:
            instruction = item['messages'][0].get('content', '')
            user = item['messages'][0].get('role', '')
            if user != 'user':
                raise ValueError(f"Unsupported data format!")
        else:
            raise ValueError(f"Unsupported data format!")
        
        
        instruction = instruction.replace("\n\n", "\n")
        prompt = generate_prompt(prompt_template, skill_definitions, instruction, tag_type)
        prompts.append(prompt)
    
        if not example_printed:
            logging.info("Displaying the first generated prompt as an example:")
            print("\n" + "="*80)
            print("--- PROMPT EXAMPLE ---")
            print(prompt)
            print("--- END OF EXAMPLE ---")
            print("="*80 + "\n")
            example_printed = True
        
    return prompts

def initialize_model(model_path: str, max_seq_length: int) -> vllm.LLM:

    logging.info(f"Initializing model from {model_path}...")
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA environment to run vLLM.")
        
    kwargs = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": torch.cuda.device_count(),
        "max_num_seqs": 512,
        "max_num_batched_tokens": max_seq_length
    }
    return vllm.LLM(**kwargs)

def run_annotation(
    client: vllm.LLM,
    dataloader: DataLoader,
    output_dir: Path,
    tag_type: str
) -> List[str]:
 
    logging.info("Starting capability annotation...")
    all_outputs = []
    
    sampling_params = vllm.SamplingParams(
        best_of=1,
        top_k=-1,
        top_p=1.0,
        temperature=0,
        stop=['<|endoftext|>'],
        max_tokens=256
    )

    try:
        for batch in tqdm(dataloader, desc=f"Annotating {tag_type}"):
            outputs = client.generate(batch, sampling_params=sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text.strip()
                all_outputs.append(generated_text)
    except Exception as e:
        logging.error(f"An error occurred during inference: {e}")
        # Save intermediate results
        intermediate_path = output_dir / f'{tag_type}_outputs_intermediate.pkl'
        logging.info(f"Saving {len(all_outputs)} generated results to {intermediate_path}")
        with intermediate_path.open('wb') as f:
            pickle.dump(all_outputs, f)
        # Re-raise the exception to halt the program
        raise e
        
    return all_outputs

def save_results(
    data_list: List[Dict[str, Any]],
    annotations: List[str],
    tag_type: str,
    output_dir: Path,
    input_filename: str
):
    """Merges annotations back into the original data and saves the result."""
    if len(data_list) != len(annotations):
        logging.warning(
            f"Data list size and annotation list size do not match! "
            f"Original data: {len(data_list)}, Annotations: {len(annotations)}. "
            "This may be due to an interruption during inference. Only successfully annotated items will be saved."
        )
        # Process only the successfully annotated part
        data_list = data_list[:len(annotations)]

    output_key = f"{tag_type}_json"
    for item, annotation in zip(data_list, annotations):
        item[output_key] = annotation

    output_filename = f"{input_filename}_{tag_type}.json"
    output_path = output_dir / output_filename

    logging.info(f"Saving annotated results to {output_path}...")
    try:
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)
        logging.info("Save successful!")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")
        # Try to save the raw outputs as a pickle backup
        pickle_path = output_dir / f'{input_filename}_{tag_type}_outputs.pkl'
        logging.info(f"Backing up raw annotation text to {pickle_path}")
        with pickle_path.open('wb') as f:
            pickle.dump(annotations, f)


def main():
    parser = argparse.ArgumentParser(description="Annotate instruction data with capability tags using a vLLM model.")
    
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the input data file to be annotated (.json or .jsonl)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the annotated results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained capability annotation model")
    parser.add_argument("--prompt_file", type=Path, required=True, help="Path to the .jsonl file containing all prompt templates")
    parser.add_argument("--cognition_skill_file", type=Path, required=True, help="Path to the cognition skills JSON file")
    parser.add_argument("--domain_skill_file", type=Path, required=True, help="Path to the domain skills JSON file")
    parser.add_argument("--task_skill_file", type=Path, required=True, help="Path to the task skills JSON file")
    parser.add_argument("--tag_type", type=str, required=True, choices=["cognition", "domain", "task"], help="The capability dimension to annotate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model")

    args = parser.parse_args()

    tag_config = {
        "domain": {
            "prompt_id": 1,
            "skill_file": args.domain_skill_file
        },
        "task": {
            "prompt_id": 2,
            "skill_file": args.task_skill_file
        },
        "cognition": {
            "prompt_id": 3,
            "skill_file": args.cognition_skill_file
        }
    }
    
    config = tag_config[args.tag_type]
    prompt_id = config["prompt_id"]
    skill_file_path = config["skill_file"]

    logging.info(f"Annotation task '{args.tag_type}' selected. Using prompt_id={prompt_id} and skill_file='{skill_file_path}'.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_list = load_data(args.data_path)
    prompt_templates = load_data(args.prompt_file)
    skill_definitions = load_data(skill_file_path)

    prompts = prepare_prompts(data_list, prompt_templates, prompt_id, skill_definitions, args.tag_type)

    dataset = AnnotationDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    client = initialize_model(args.model_path, args.max_seq_length)

    annotations = run_annotation(client, dataloader, args.output_dir, args.tag_type)

    input_filename_stem = args.data_path.stem
    save_results(data_list, annotations, args.tag_type, args.output_dir, input_filename_stem)

if __name__ == "__main__":
    main()