import json
import os
import re
import argparse
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path

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

def extract_tags(raw_annotations: List[str], tag_type: str) -> List[List[str]]:
    """
    Extracts tags from a list of raw LLM annotation strings using regex.
    """
    if tag_type == 'cognition':
        pattern = r'"skill":\s*"(.*?)"'
    elif tag_type == 'domain':
        pattern = r'"domain":\s*"(.*?)"'
    else: # task
        pattern = r'"task":\s*"(.*?)"'
    
    extracted_lists = []
    for text in raw_annotations:
        
        text = text.replace("'", '"')
        matches = re.findall(pattern, text)
        extracted_lists.append([m.replace('"', '') for m in matches])
    return extracted_lists

def clean_and_count_tags(current_tags: List[str], valid_skills: Set[str]) -> Tuple[List[str], int]:
    """
    Cleans a list of tags against a set of valid skills and counts the invalid ones.
    """
    clean_tags_list = []
    invalid_count = 0
    for tag in current_tags:
        if tag in valid_skills:
            clean_tags_list.append(tag)
        else:
            invalid_count += 1
    return clean_tags_list, invalid_count


def main():
    parser = argparse.ArgumentParser(description="Post-process annotated data to clean and validate tags.")
    
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the annotated data file (.json or .jsonl).")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to save the cleaned output JSON file.")
    parser.add_argument("--tag_type", type=str, required=True, choices=["cognition", "domain", "task"], help="The capability dimension to process.")
    
    # Arguments for skill definition files
    parser.add_argument("--cognition_skill_file", type=Path, required=True, help="Path to the cognition skills JSON file.")
    parser.add_argument("--domain_skill_file", type=Path, required=True, help="Path to the domain skills JSON file.")
    parser.add_argument("--task_skill_file", type=Path, required=True, help="Path to the task skills JSON file.")
    
    args = parser.parse_args()

    # Select the correct skill file based on the tag_type
    skill_file_map = {
        "cognition": args.cognition_skill_file,
        "domain": args.domain_skill_file,
        "task": args.task_skill_file,
    }
    skill_file = skill_file_map[args.tag_type]

    data = load_json_data(args.data_path)
    skill_definitions = load_json_data(skill_file)
    valid_skills_set = get_valid_skills(skill_definitions, args.tag_type)

    json_key = f"{args.tag_type}_json"
    raw_annotations = [item.get(json_key, "") for item in data]
    extracted_tags = extract_tags(raw_annotations, args.tag_type)
    
    # Clean invalid tags
    total_invalid_count = 0
    for item, tags_for_item in zip(data, extracted_tags):
        cleaned_tags, num_invalid = clean_and_count_tags(tags_for_item, valid_skills_set)
        
        item[json_key] = cleaned_tags
        total_invalid_count += num_invalid

    print(f"Number of Wrong Tag: {total_invalid_count}")

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Cleaned data saved to {args.output_file}")

if __name__ == '__main__':
    main()