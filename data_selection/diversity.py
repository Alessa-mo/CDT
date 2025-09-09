import json
import math
import random
import argparse
import os
import copy
from typing import List, Dict, Any, Tuple

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads data from a .json or .jsonl file."""
    file_path = os.path.expanduser(file_path)
    file_extension = file_path.split('.')[-1]
    if file_extension == "jsonl":
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def group_data_by_capability(data: List[Dict[str, Any]]) -> Dict[Tuple, Dict]:
    """
    Groups data by capability tags
    """
    print("Grouping data by capability tags...")
    four_tags = {}
    three_tags = {}
    other_tags = {}  # This dict is used for filtering invalid/unwanted tags

    for idx, item in enumerate(data):
        tags = item.get("tag", {})
        cognitions = tags.get("cognition", [])
        domain = tags.get("domain", [])
        task = tags.get("task", [])

        # Items with missing domain/task tags are filtered out
        domain_val = domain[0] if domain else -1
        task_val = task[0] if task else -1
        
        target_dict = None
        tag_comb = None

        # Handle the case of two cognition tags
        if len(cognitions) >= 2:
            # Sort cognition tags to treat (c1, c2) and (c2, c1) as the same capability
            c1, c2 = sorted(cognitions[:2])
            tag_comb = (c1, c2, domain_val, task_val)
            target_dict = four_tags
        # Handle the case of one cognition tags
        elif len(cognitions) == 1:
            tag_comb = (cognitions[0], domain_val, task_val)
            target_dict = three_tags
        else: # 0 cognition tags, filter out
            tag_comb = (domain_val, task_val)
            target_dict = other_tags
        
        # Filter items with missing domain/task
        if -1 in tag_comb:
            target_dict = other_tags

        if tag_comb not in target_dict:
            target_dict[tag_comb] = {'number': 1, 'index': []}
        target_dict[tag_comb]['number'] += 1
        target_dict[tag_comb]['index'].append(idx)

    sorted_four = dict(sorted(four_tags.items(), key=lambda item: item[1]['number'], reverse=True))
    sorted_three = dict(sorted(three_tags.items(), key=lambda item: item[1]['number'], reverse=True))

    # Merge, implicitly discarding other_tags
    return {**sorted_four, **sorted_three}

def select_diverse_data(data: List[Dict[str, Any]], capability_groups: Dict[Tuple, Dict], num_to_select: int) -> List[Dict[str, Any]]:
    """
    Selects a subset of data by maximizing capability diversity.
    """
    selected_indices = []
    
    groups_copy = copy.deepcopy(capability_groups)
    
    # Sufficient diversity (sample one from each unique group)
    if len(groups_copy) >= num_to_select:
        print("Sufficient capability diversity. Selecting one sample per unique group...")
        for group_info in groups_copy.values():
            indices = group_info['index']
            if not indices: continue
            
            selected_idx = random.sample(indices, 1)[0]
            selected_indices.append(selected_idx)
            if len(selected_indices) == num_to_select:
                break
    
    # Insufficient Diversity (iteratively sample from all groups)
    else:
        print("Insufficient capability diversity. Performing iterative, stratified sampling...")
        iteration = 0
        while len(selected_indices) < num_to_select:
            iteration += 1
            print(f"  > Iteration {iteration}...")
            
            num_selected_in_iteration = 0
            for group_info in groups_copy.values():
                indices = group_info['index']
                if not indices: continue

                selected_idx = random.sample(indices, 1)[0]
                selected_indices.append(selected_idx)
                indices.remove(selected_idx)
                num_selected_in_iteration += 1
                
                if len(selected_indices) == num_to_select:
                    break
            
            # If an iteration selected zero items, all groups are exhausted.
            if num_selected_in_iteration == 0:
                print("Warning: All available data points have been selected, but target number not reached.")
                break
                
    return [data[i] for i in selected_indices]


def main():
    parser = argparse.ArgumentParser(description="Perform diversity-driven data selection based on CDT capability tags.")
    parser.add_argument("--data_path", required=True, help="Path to the tagged data pool file.")
    parser.add_argument("--output_file", required=True, help="Path to save the selected data file.")
    parser.add_argument("--top_p", type=float, default=0.2, help="Proportion of the total data to select (e.g., 0.2 for 20%).")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Loading data from {args.data_path}...")
    data = load_json_data(args.data_path)
    random.shuffle(data)
    
    num_to_select = math.ceil(args.top_p * len(data))
    print(f"Total data points: {len(data)}. Target selection: {num_to_select} ({args.top_p * 100:.1f}%)")
    
    capability_groups = group_data_by_capability(data)
    print(f"Found {len(capability_groups)} unique capability groups.")
    
    selected_data = select_diverse_data(data, capability_groups, num_to_select)
    
    print(f"\nSuccessfully selected {len(selected_data)} data points.")
    print(f"Saving selected data to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=2)
    print("Done.")

if __name__ == '__main__':
    main()