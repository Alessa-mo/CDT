from torch.utils.data import Dataset
import random

def gen_prompt(prompt_jsons, prompt_id, skill_jsons, instruction, tag):
    prompt_json = prompt_jsons[prompt_id-1]
    assert prompt_json['prompt_id'] == prompt_id

    sys_prompt = prompt_json['system_prompt']
    prompt_template = prompt_json['prompt_template']
    defaults = prompt_json['defaults']
    skills =""

    skill2defs = {}
    for name, definitions in skill_jsons.items():
        skill2defs[name] = definitions

    skill_name = list(skill2defs.keys())
    random.shuffle(skill_name)

    if tag == "domain":
        index = 1
        for name in skill_name:
            lists = skill2defs[name]
            skills+=f"\n{index}. {name}: "
            skills+=(', ').join(lists)
            index+= 1
    else: 
        for name in skill_name:
            definition = skill2defs[name]
            skills += f"\n{name}: {definition}"

    prompt = prompt_template.format(question=instruction, skill=skills, **defaults)
    return sys_prompt+' '+prompt


class SkillTagDataset(Dataset):
    def __init__(self, prompt_id, instructions, inputs, prompt_jsons, skill_jsons, tag):
        
        for idx, instruction, input in zip(range(len(instructions)), instructions, inputs):
            if not input == '':
                instructions[idx] = instruction + '\n' + 'Input: ' + input

        texts = []
        for instruction in instructions:
            instruction = instruction.replace("\n\n", "\n")
            text = gen_prompt(prompt_jsons, prompt_id, skill_jsons, instruction, tag)
            texts.append(text)
        self.input = texts

    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        return self.input[idx]


    