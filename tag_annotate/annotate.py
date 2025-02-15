import torch
import os
import tqdm
import pickle
from torch.utils.data import DataLoader
import json
import os
from tag_dataset import SkillTagDataset
import vllm
def getSkills(skill_jsons, tag):
    all_skills = []
    if tag == "domain":
        for name, lists in skill_jsons.items():
            all_skills += lists
    else:
        for name in skill_jsons.keys():
            all_skills.append(name)

    return all_skills

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


def readAlpacaInstructions(data):
    instructions = []
    inputs = []
    for item in data:
        instructions.append(item['instruction'])
        inputs.append(item['input'])
    
    assert len(instructions) == len(inputs)
    return instructions, inputs

def readTuluInstructions(data):
    instructions = []
    inputs = []
    for item in data:
        messages = item['messages']
        instructions.append(messages[0]['content'])
        inputs.append('')
        if messages[0]['role'] != 'user':
            raise Exception('format error!')
    assert len(instructions) == len(inputs)
    return instructions, inputs

def tagDomainTask(client, dataloader, output_path, tag_type):

    all_outputs = []
    try:
        for input in tqdm.tqdm(dataloader, desc='Generating responses'):
            outputs = client.generate(input, sampling_params=vllm.SamplingParams(
                best_of=1,
                top_k=-1,
                top_p=1.0,
                temperature=0,
                stop=['<|endoftext|>'],
                max_tokens=256
            ))
            for output in outputs:
                generated_text = output.outputs[0].text
                all_outputs.append(generated_text)
    except:
        pickle.dump(all_outputs, open(output_path + '/' + tag_type + '_outputs.pkl', 'wb'))
    
    return all_outputs

# parameter
data_type = 'Tulu'
tag = "domain" #choose from cogition domain task
data_path = 'path/to/data'
output_path = 'output_dir'
batch_size = 32

# fixed
max_seq_length = 2048
prompt_file = 'prompt/anotation_prompt.jsonl'


file_path = {
    "domain":{
                "skill_file": "prompt/domain.json",
                "prompt_id": 1,
                "cpt_path": 'path_to_domain_tag_model'
            },
    "task":{
            "skill_file": "prompt/task.json",
            "prompt_id": 2,
            "cpt_path": 'path_to_task_tag_model'
            },
    "cognition":{
            "skill_file": "prompt/cogntion.json",
            "prompt_id": 3,
            "cpt_path": 'path_to_cognition_tag_model'
            }
    }

prompt_id = file_path[tag]['prompt_id']
skillset_file = file_path[tag]['skill_file']
cpt_path = file_path[tag]['cpt_path']

prompt_jsons = getJsonList(prompt_file)
skill_jsons = getJsonList(skillset_file)
data_jsons = getJsonList(data_path)
print(len(data_jsons))
if data_type == 'Tulu':
    instructions, inputs = readTuluInstructions(data_jsons)
else:
    instructions, inputs = readAlpacaInstructions(data_jsons)


dataset = SkillTagDataset(prompt_id, instructions, inputs, prompt_jsons, skill_jsons, tag)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


kwargs = {
    "model": cpt_path,
    "tokenizer": cpt_path,
    "trust_remote_code": True,
    "max_num_seqs": 512,
    "tensor_parallel_size": torch.cuda.device_count(),
    "max_num_batched_tokens": max_seq_length
}
client = vllm.LLM(**kwargs)

all_skills = getSkills(skill_jsons, tag)
all_outputs = tagDomainTask(client, dataloader, output_path,tag)
print(len(all_outputs))
print(len(data_jsons))
assert len(all_outputs) == len(data_jsons)

skill_type = tag + "_json"
file_name = os.path.basename(data_path)
file_name = os.path.splitext(file_name)[0]

for idx, item in enumerate(data_jsons):
    data_jsons[idx][skill_type] = all_outputs[idx]

try:
    json_data = json.dumps(data_jsons, ensure_ascii=False, indent=4)
    with open(output_path +'/' + file_name + '_' + tag + '.json', 'w', encoding='utf-8') as f:
        f.write(json_data)
except:
    pickle.dump(all_outputs, open(output_path + '/' + file_name + '_' + tag + '_outputs.pkl', 'wb'))

