import json
import os
import re
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

def extract_skill(answers, task_type):
    if task_type == 'cognition':
        pattern = r'"skill": "(.*?)"'
    elif task_type == 'domain':
        pattern = r'"domain": "(.*?)"'
    else:
        pattern = r'"task": "(.*?)"'
    print(len(answers))
    skill_strings = []
    for idx, answer in enumerate(answers):
        answer = answer.replace("'", '"')
        
        matches = re.findall(pattern, answer)
        skills = []
        for i in matches:
            skills.append(i.replace('"', ''))
        skill_strings.append(skills)
    return skill_strings

def cleanTag(all_skills, cur_tag):
    global cnt
    clean_tag = []
    for skill in cur_tag:
        if skill in all_skills:
            clean_tag.append(skill)
        else:
            cnt += 1
    return clean_tag

# parameter
data_path = 'path/to/data'
type = "task" #choose from cogition domain task
output_file = 'output_file'

# fixed
file_path = {
    "domain": "prompt/domain.json", 
    "task":"prompt/task.json",
    "cognition": "prompt/cogntion.json",
    }
skill_file = file_path[type]

data = getJsonList(data_path)
all_skills_json = getJsonList(skill_file)
all_skills = getSkills(all_skills_json, type)

tag_skill_json = [item[type+'_json'] for item in data]
tag_skills = extract_skill(tag_skill_json, type)
cnt = 0
for item, tag_skill in zip(data, tag_skills):
    tag_skill = cleanTag(all_skills, tag_skill)
    item[type+'_json'] = tag_skill


print("Number of Wrong Tag: ",cnt)

json_data = json.dumps(data, ensure_ascii=False, indent=2)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_data)