import json
import math
from collections import Counter
import os

def getJsonList(file_path):
    file_path = os.path.expanduser(file_path)
    file_extension = file_path.split('.')[-1]
    if file_extension=="jsonl":
        with open(file_path, 'r') as f:
            json_list = []
            for line in f:
                json_list.append(json.loads(line))
            return json_list
    else:
        with open(file_path, 'r') as f:
            return json.load(f)
        
file_path = "path/to/labeled/data"
raw_data = getJsonList(file_path)

NUM_COGNITION = 18
NUM_DOMAIN = 33
NUM_TASK = 16

quadruplet_counter = Counter()
unique_quadruplets = set()

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
        c1, c2 = sorted(cognition[:2])  # Two cognition labels

    # Update counters
    quad = (c1, c2, domain, task)
    quadruplet_counter[quad] += 1
    unique_quadruplets.add(quad)

    data.append(item)

def compute_entropy(counter):
    total = sum(counter.values())
    entropy = -sum((count / total) * math.log(count / total, 2) for count in counter.values())
    return entropy

# Calculate Balance metrics
quadruplet_entropy = compute_entropy(quadruplet_counter)

# Composite Coverage
MAX_COG_PAIR = (NUM_COGNITION * (NUM_COGNITION - 1)) // 2 + NUM_COGNITION
composite_coverage = len(unique_quadruplets) / (MAX_COG_PAIR * NUM_DOMAIN * NUM_TASK)

results = {
    "Balance": quadruplet_entropy,
    "Coverage": composite_coverage
}
print(results)
