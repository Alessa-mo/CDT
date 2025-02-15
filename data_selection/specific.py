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
    def add_to_dic(tag_dict, cur_tag, data_index):
        if cur_tag in tag_dict.keys():
            tag_dict[cur_tag]['number'] += 1
            tag_dict[cur_tag]['index'].append(data_index)
        else:
            tag_dict[cur_tag] = {'number': 1, 'index': [data_index]}
            
    three_dims = {}
    two_dims = {}
    one_dims = {}
    for idx, item in enumerate(data):
        tags = item['tag']
        cognitions = tags['cognition']
        cognitions = sorted(cognitions)
        domain = tags['domain'][0] if len(tags['domain']) > 0 else -1
        task = tags['task'][0] if len(tags['task']) > 0 else -1
        if len(cognitions) > 0 and len(tags['domain']) > 0 and len(tags['task']) > 0 :
            domain = tags['domain'][0]
            task = tags['task'][0]
            if len(cognitions) == 1:
                three_comb = (cognitions[0], domain, task)
                add_to_dic(three_dims, three_comb, idx)
                add_to_dic(two_dims, (cognitions[0],domain), idx)
                add_to_dic(two_dims, (cognitions[0],task), idx)
                add_to_dic(two_dims, (domain, task), idx)
                add_to_dic(one_dims, (cognitions[0]), idx)
            else:
                four_comb = (cognitions[0], cognitions[1], domain, task)
                add_to_dic(three_dims, four_comb, idx)
                add_to_dic(two_dims, (cognitions[0], cognitions[1], domain), idx)
                add_to_dic(two_dims, (cognitions[0], cognitions[1], task), idx)
                add_to_dic(two_dims, (domain, task), idx)
                add_to_dic(one_dims, (cognitions[0], cognitions[1]), idx)
            add_to_dic(one_dims, domain, idx)
            add_to_dic(one_dims, task, idx)
            
    three_dims = dict(sorted(three_dims.items(), key=lambda x: x[1]['number'], reverse=True))
    two_dims = dict(sorted(two_dims.items(), key=lambda x: x[1]['number'], reverse=True))
    one_dims = dict(sorted(one_dims.items(), key=lambda x: x[1]['number'], reverse=True))
    return [three_dims, two_dims, one_dims]
    

def simpleSelect(valid_all_tags, train_all_tags, number, selected_idx):
    train_tags_for_valid = {}
    for comb in valid_all_tags.keys():
        try:
            train_tags_for_valid[comb]=train_all_tags[comb]
        except:
            print("No such combination in train file, ", comb)
    train_tags_for_valid = dict(sorted(train_tags_for_valid.items(), key=lambda x: x[1]['number'], reverse=True))
    
    if len(train_tags_for_valid) >= number:
        print("Enough tag combination. Filter by proportion")
        for v in train_tags_for_valid.values():
            cur_tag_idx = v['index']
            to_select_idx = set(cur_tag_idx).difference(selected_idx)
            select_idx = random.sample(list(to_select_idx), 1)
            assert len(select_idx) == 1
            select_idx = select_idx[0]
            selected_idx.add(select_idx)
            if len(selected_idx) == number:
                break
    else:
        print("Not enough tag combination. select multi iterations")
        while len(selected_idx) < number and len(train_tags_for_valid) > 0:
            remove_k = []
            for k, v in train_tags_for_valid.items():
                cur_tag_idx = v['index']
                to_select_idx = set(cur_tag_idx).difference(selected_idx)
                if len(to_select_idx) == 0:
                    remove_k.append(k)
                    continue
                select_idx = random.sample(list(to_select_idx), 1)
                assert len(select_idx) == 1
                select_idx = select_idx[0]
                cur_tag_idx.remove(select_idx)
                selected_idx.add(select_idx)
                if len(cur_tag_idx) == 0:
                    print('remove: ',k )
                    remove_k.append(k)
                else:
                    train_tags_for_valid[k]['index'] = cur_tag_idx
                    train_tags_for_valid[k]['number'] -= 1
                if len(selected_idx) == number:
                    break
            for k in remove_k:
                del train_tags_for_valid[k]
    return selected_idx  

def select(data, valid_all_tags, train_all_tags, number):
    selected_idx = set()
    cnt = 0
    for valid_dim_dic, train_dim_dic in zip(valid_all_tags, train_all_tags):
        print("cnt. ", cnt)
        selected_idx = simpleSelect(valid_dim_dic, train_dim_dic, number, selected_idx)
        if len(selected_idx) == number:
            break
        cnt += 1
        
    if len(selected_idx) < number:
        print('random select, ', number-len(selected_idx))
        all_random_idx = [idx for idx in range(len(data)) if idx not in selected_idx]
        random_idx = random.sample(all_random_idx, number-len(selected_idx))
        selected_idx = list(selected_idx) + random_idx
    return [data[i] for i in selected_idx]
    

data_path = 'path_to_tagged_data_pool'
valid_path = 'path_to_tagged_validation_data'
top_p = 0.2 # selection propotion
output_file = 'path_to_output'
seed = 1234

random.seed(seed)
train_data = getJsonList(data_path)
valid_data = getJsonList(valid_path)
number = math.ceil(top_p*len(train_data))
all_valid_tags = computeTag(valid_data)
all_train_tags = computeTag(train_data)
selected_data = select(train_data, all_valid_tags, all_train_tags, number)

json_data = json.dumps(selected_data, ensure_ascii=False, indent=2)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_data)
