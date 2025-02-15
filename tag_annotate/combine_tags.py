import json
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

def getSkills(skill_jsons, tag):
    all_skills = []
    if tag == "domain":
        for name, lists in skill_jsons.items():
            all_skills += lists
    else:
        for name in skill_jsons.keys():
            all_skills.append(name)
    return all_skills

# parameter
cognition_path = 'path_to_cleaned_cognition_tag'
domain_path = "path_to_cleaned_domain_tag"
task_path = "path_to_cleaned_task_tag"
output_file = "path_to_combined_tag"

# fixed
domain_file = "prompt/domain.json"
task_file = "prompt/task.json"
cognition_file = "prompt/cogntion.json"

cognition_data = getJsonList(cognition_path)
domain_data = getJsonList(domain_path)
task_data = getJsonList(task_path)
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
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_data)