
import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

TagKey = Tuple[Any, ...]      # A composite capability key, e.g., (c,d,t), (c,d), (c)
TagInfo = Dict[str, Any]     # Stores the count and indices for a key: {"number": int, "index": List[int]}
TagDict = Dict[TagKey, TagInfo] # A dictionary mapping capability keys to their info


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


def _add_to_dict(tag_dict: TagDict, key: TagKey, data_index: int) -> None:
    if key in tag_dict:
        tag_dict[key]['number'] += 1
        tag_dict[key]['index'].append(data_index)
    else:
        tag_dict[key] = {'number': 1, 'index': [data_index]}


def _sorted_by_freq(tag_dict: TagDict) -> TagDict:
    return dict(sorted(tag_dict.items(), key=lambda x: x[1]['number'], reverse=True))


def analyze_dataset_capabilities(data: List[Dict[str, Any]]) -> List[TagDict]:
    """
    Analyzes a dataset to extract capability composites at 3 levels:
    3-dim, 2-dim, and 1-dim
    """
    three_dims: TagDict = {}
    two_dims: TagDict = {}
    one_dims: TagDict = {}

    for idx, item in enumerate(data):
        tags = item['tag']
        # Sort cognition tags to treat (c1, c2) and (c2, c1) as the same capability
        cognitions: List[Any] = sorted(tags['cognition'])
        domains: List[Any] = tags['domain']
        tasks: List[Any] = tags['task']

        # Only proceed for data points that have tags in all three dimensions
        if len(cognitions) > 0 and len(domains) > 0 and len(tasks) > 0:
            d = domains[0]
            t = tasks[0]

            # Handle the case of a single cognition tag
            if len(cognitions) == 1:
                c = cognitions[0]
                # Add composites for all three levels
                _add_to_dict(three_dims, (c, d, t), idx)
                _add_to_dict(two_dims, (c, d), idx)
                _add_to_dict(two_dims, (c, t), idx)
                _add_to_dict(two_dims, (d, t), idx)
                _add_to_dict(one_dims, c, idx)
            
            # Handle the case of two cognition tags
            else:
                c1, c2 = cognitions[0], cognitions[1]
                # Add composites for all three levels
                _add_to_dict(three_dims, (c1, c2, d, t), idx)
                _add_to_dict(two_dims, (c1, c2, d), idx)
                _add_to_dict(two_dims, (c1, c2, t), idx)
                _add_to_dict(two_dims, (d, t), idx)
                _add_to_dict(one_dims, (c1, c2), idx)

            _add_to_dict(one_dims, d, idx)
            _add_to_dict(one_dims, t, idx)

    return [_sorted_by_freq(three_dims), _sorted_by_freq(two_dims), _sorted_by_freq(one_dims)]


def _select_from_one_dimension(
    valid_groups: TagDict,
    train_groups: TagDict,
    target_count: int,
    selected_indices: set[int],
) -> set[int]:
    """
    Performs one level of selection.
    """
    # Find training groups that match a capability in the validation set
    train_groups_for_valid: TagDict = {}
    for capability_key in valid_groups.keys():
        if capability_key in train_groups:
            train_groups_for_valid[capability_key] = train_groups[capability_key]
        else:
            print(f"INFO: Capability {capability_key} from validation set not found in training pool.")

    train_groups_for_valid = _sorted_by_freq(train_groups_for_valid)
    
    # Enough groups
    if len(train_groups_for_valid) >= target_count:
        for group_info in train_groups_for_valid.values():
            cur_tag_idx = group_info['index']
            # Find indices in this group that have not already been selected
            to_select = set(cur_tag_idx).difference(selected_indices)
            select_idx = random.sample(list(to_select), 1)
            assert len(select_idx) == 1
            select_idx = select_idx[0]
            selected_indices.add(select_idx)
            if len(selected_indices) == target_count:
                break
    
    # Groups are not diverse enough.
    else:
        # This loop continues as long as there's work to do and quota isn't met.
        while len(selected_indices) < target_count and len(train_groups_for_valid) > 0:
            keys_to_remove: List[TagKey] = []
            
            for key, group_info in list(train_groups_for_valid.items()):
                cur_tag_idx = group_info['index']
                # Find indices in this group that have not already been selected
                to_select = set(cur_tag_idx).difference(selected_indices)
                # If all indices for this capability have been used, mark it for removal
                if len(to_select) == 0:
                    keys_to_remove.append(key)
                    continue
                select_idx = random.sample(list(to_select), 1)
                assert len(select_idx) == 1
                select_idx = select_idx[0]
                # Remove the selected index to prevent it from being chosen again by this same group in the next iteration
                cur_tag_idx.remove(select_idx)
                selected_indices.add(select_idx)
                if len(cur_tag_idx) == 0:
                    keys_to_remove.append(key)
                else:
                    train_groups_for_valid[key]['index'] = cur_tag_idx
                    train_groups_for_valid[key]['number'] -= 1
                if len(selected_indices) == target_count:
                    break
            for key in keys_to_remove:
                del train_groups_for_valid[key]

    return selected_indices


def hierarchical_capability_selection(
    train_data: List[Dict[str, Any]],
    valid_capability_sets: List[TagDict],
    train_capability_sets: List[TagDict],
    num_to_select: int,
) -> List[Dict[str, Any]]:
    """
    Orchestrates the hierarchical selection process: 3-dim -> 2-dim -> 1-dim -> random fill.
    """
    selected_indices: set[int] = set()
    dimension_names = ["3-dim", "2-dim", "1-dim"]

    # Iterate through the capability dimensions, from highest (3D) to lowest (1D)
    for i, (valid_groups, train_groups) in enumerate(zip(valid_capability_sets, train_capability_sets)):
        dimension_name = dimension_names[i]
        print(f"--- Starting selection with {dimension_name} capabilities ---")
        
        selected_indices = _select_from_one_dimension(valid_groups, train_groups, num_to_select, selected_indices)
        
        print(f"Selected {len(selected_indices)}/{num_to_select} items so far.")
        
        # If the selection quota is met, exit the hierarchical process early
        if len(selected_indices) == num_to_select:
            break

    # If quota is not met after all hierarchical selections, backfill with random samples.
    if len(selected_indices) < num_to_select:
        num_to_fill = num_to_select - len(selected_indices)
        print(f"\nTarget not met. Randomly selecting {num_to_fill} more items...")
        
        # Find all indices that have not yet been selected
        available_indices = [idx for idx in range(len(train_data)) if idx not in selected_indices]
        if len(available_indices) < num_to_fill:
            print("Warning: All available data points have been selected, but target number not reached.")
            selected_indice = list(selected_indice) + available_indices
        else:
            random_indices = random.sample(available_indices, num_to_fill)
            selected_indice = list(selected_indice) + random_indices

    return [train_data[i] for i in selected_indices]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CDT capability-oriented data selection")
    p.add_argument("--train_file", required=True, help="Path to the tagged training data pool.")
    p.add_argument("--valid_file", required=True, help="Path to the tagged validation set.")
    p.add_argument("--output_file", required=True, help="Path to save the selected data file.")
    p.add_argument("--top_p", type=float, default=0.2, help="Proportion of the total data to select (e.g., 0.2 for 20%).")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)

    print(f"Loading training data from: {args.train_file}")
    train_data = load_json_data(args.train_file)
    print(f"Loading validation data from: {args.valid_file}")
    valid_data = load_json_data(args.valid_file)
    
    num_to_select = math.ceil(args.top_p * len(train_data))
    print(f"\nTotal training data: {len(train_data)}. Target selection: {num_to_select} ({args.top_p * 100:.1f}%)")

    print("\nAnalyzing capabilities of validation set...")
    valid_capability_sets = analyze_dataset_capabilities(valid_data)
    print("Analyzing capabilities of training set...")
    train_capability_sets = analyze_dataset_capabilities(train_data)

    selected_data = hierarchical_capability_selection(train_data, valid_capability_sets, train_capability_sets, num_to_select)

    print(f"\nSuccessfully selected {len(selected_data)} data points.")
    print(f"Saving selected data to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=1)
    print("Done.")


if __name__ == "__main__":
    main()