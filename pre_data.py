import json

def read_json(path):
    with open(path,'r',encoding='utf-8') as file:
        all_data = []
        for line in file:
            data = json.loads(line)
            all_data.append(data)
    return all_data