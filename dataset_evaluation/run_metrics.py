import json
import math
from collections import Counter
import os
from typing import List, Dict, Any, Counter as CounterType
import argparse
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
        
def compute_entropy(counter: CounterType) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total, 2) for count in counter.values())
    return entropy

def main():
    parser = argparse.ArgumentParser(description="Compute Balance and Coverage metrics.")   
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the tagged data file (.json or .jsonl).")
    args = parser.parse_args()
    
    raw_data = load_json_data(args.data_path)
    
    NUM_COGNITION = 18
    NUM_DOMAIN = 33
    NUM_TASK = 16

    capability_counter = Counter()
    unique_capabilities = set()
    # Filter and process data
    data = []
    for item in raw_data:
        tags = item.get("tag", {})
        cognition = tags.get("cognition", [])
        domain = tags.get("domain", [])
        task = tags.get("task", [])

        # Filter out invalid items
        if not cognition or not domain or not task:
            continue

        domain = domain[0]
        task = task[0]

        # Normalize cognition
        if len(cognition) == 1: # One cognition label
            c1, c2 = cognition[0], "0"
        else:
            # Two cognition labels, sort cognition tags to treat (c1, c2) and (c2, c1) as the same capability
            c1, c2 = sorted(cognition[:2])

        # Update counters
        quad = (c1, c2, domain, task)
        capability_counter[quad] += 1
        unique_capabilities.add(quad)

        data.append(item)
        
    # Calculate Balance metrics
    capability_entropy = compute_entropy(capability_counter)

    # Composite Coverage
    MAX_COG_PAIR = (NUM_COGNITION * (NUM_COGNITION - 1)) // 2 + NUM_COGNITION
    composite_coverage = len(unique_capabilities) / (MAX_COG_PAIR * NUM_DOMAIN * NUM_TASK)

    results = {
        "Balance(bits)": capability_entropy,
        "Coverage(%)": composite_coverage * 100
    }
    print(results)

if __name__ == '__main__':
    main()