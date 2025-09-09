import json
import os
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
import argparse

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    file_path = os.path.expanduser(file_path)
    file_extension = file_path.split('.')[-1]
    if file_extension == "jsonl":
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def get_valid_skills(skill_definitions: Dict[str, Any], tag_type: str) -> Set[str]:
    """
    Extracts all valid skill names into a set.
    """
    all_skills = []
    if tag_type == "domain":
        for name, lists in skill_definitions.items():
            all_skills.extend(lists)
    else:
        for name in skill_definitions.keys():
            all_skills.append(name)
    return set(all_skills)

def main():
    parser = argparse.ArgumentParser(description="Merge all tags into one file.")
    
    parser.add_argument("--cognition_path", type=Path, required=True, help="Path to the cleaned cognition tag file (.json or .jsonl).")
    parser.add_argument("--domain_path", type=Path, required=True, help="Path to the cleaned domain tag file (.json or .jsonl).")
    parser.add_argument("--task_path", type=Path, required=True, help="Path to the cleaned task tag file (.json or .jsonl).")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to save the final tag JSON file.")
    parser.add_argument("--cognition_skill_file", type=Path, required=True, help="Path to the cognition skills JSON file.")
    parser.add_argument("--domain_skill_file", type=Path, required=True, help="Path to the domain skills JSON file.")
    parser.add_argument("--task_skill_file", type=Path, required=True, help="Path to the task skills JSON file.")
    
    args = parser.parse_args()
    cognition_data = load_json_data(args.cognition_path)
    domain_data = load_json_data(args.domain_path)
    task_data = load_json_data(args.task_path)  
    assert len(cognition_data) == len(domain_data) == len(task_data)

    for i in range(len(cognition_data)):
        tags = {}
        cognitions = cognition_data[i]['cognition_json']
        domains = domain_data[i]['domain_json']
        tasks = task_data[i]['task_json']
        tags['cognition'] = cognitions
        tags['task'] = tasks
        tags['domain'] = domains

        del cognition_data[i]['cognition_json']

        cognition_data[i]['tag'] = tags
    json_data = json.dumps(cognition_data, ensure_ascii=False, indent=2)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(json_data)
        
    print(f"Final data saved to {args.output_file}")
if __name__ == '__main__':
    main()