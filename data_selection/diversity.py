import json
import os
import math
import random

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

def computeTag(data):
    four_tags = {}
    three_tags = {}
    other_tags = {}
    for idx, item in enumerate(data):
        tags = item['tag']
        cognitions = tags['cognition']
        cognitions = sorted(cognitions)
        domain = tags['domain'][0] if len(tags['domain']) > 0 else -1
        task = tags['task'][0] if len(tags['task']) > 0 else -1
        if len(cognitions) < 2:
            if len(cognitions) == 0:
                tag_comb = (domain, task)
                tag_dict = other_tags
            else:
                tag_comb = (cognitions[0], domain, task)
                tag_dict = three_tags
        else:
            tag_comb = (cognitions[0],cognitions[1], domain, task)
            tag_dict = four_tags
        if -1 in tag_comb:
            tag_dict = other_tags
        if tag_comb in tag_dict.keys():
            tag_dict[tag_comb]['number'] += 1
            tag_dict[tag_comb]['index'].append(idx)
        else:
            tag_dict[tag_comb] = {'number': 1, 'index': [idx]}
    four_tags = dict(sorted(four_tags.items(), key=lambda x: x[1]['number'], reverse=True))
    three_tags = dict(sorted(three_tags.items(), key=lambda x: x[1]['number'], reverse=True))
    return four_tags | three_tags

def simpleSelect(data, all_tags, number):
    selected_idx = []
    if len(all_tags) >= number:
        print("Enough tag combination. Filter by proportion")
        for v in all_tags.values():
            cur_tag_idx = v['index']
            select_idx = random.sample(cur_tag_idx, 1)
            assert len(select_idx) == 1
            select_idx = select_idx[0]
            selected_idx.append(select_idx)
            if len(selected_idx) == number:
                break
    else:
        print("Not enough tag combination. select multi iterations")
        cnt = 0
        while len(selected_idx) < number:
            cnt += 1
            print("Try cnt: ", cnt)
            for k, v in all_tags.items():
                cur_tag_idx = v['index']
                if len(cur_tag_idx) == 0:
                    continue
                select_idx = random.sample(cur_tag_idx, 1)
                assert len(select_idx) == 1
                select_idx = select_idx[0]
                cur_tag_idx.remove(select_idx)
                selected_idx.append(select_idx)
                all_tags[k]['index'] = cur_tag_idx
                all_tags[k]['number'] -= 1
                if len(selected_idx) == number:
                    break
    return [data[i] for i in selected_idx]   
    

data_path = 'path_to_tagged_data_pool'
top_p = 0.2 # selection propotion
output_path = 'path_to_output'
output_file = 'path_to_output'
seed = 1234

random.seed(seed)
data = getJsonList(data_path)
number = math.ceil(top_p*len(data))
all_tags = computeTag(data)
selected_data = simpleSelect(data, all_tags, number)

json_data = json.dumps(selected_data, ensure_ascii=False, indent=2)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_data)
