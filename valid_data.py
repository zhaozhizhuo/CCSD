import json
import pre_data
p = ['train','test','dev']
label = []
for pp in p:
    path = './data_preprocess/{}.json'.format(pp)
    contents = pre_data.read_json(path)
    userid = {}
    for content in contents:
        user_id = content['user_id']
        if user_id in userid:
            userid[user_id] += int(1)
        else:
            userid[user_id] = int(1)
        label.append(content['norm_syndrome'])

    x = userid.items()
    keep_user = {k for k, v in userid.items() if v >= int(2)}
    keep_label = {}
    for content in contents:
        user_id = content['user_id']
        if user_id in keep_user and user_id in keep_label:
            keep_label[user_id] = keep_label[user_id] + '|' + content['norm_syndrome']
        if user_id in keep_user and user_id not in keep_label:
            keep_label[user_id] = content['norm_syndrome']
    print(len(keep_user))

    single_userid = []
    #更改数据集，因为现在的数据集是但标签
    with open('./data_preprocess/new_{}.json'.format(pp),'w',encoding='utf-8',newline='') as json_file:
        for content in contents:
            user_id = content['user_id']
            single_userid.append(user_id)
            if user_id in keep_label and user_id not in single_userid:
                content['norm_syndrome'] = keep_label[user_id]
            json.dump(content,json_file, ensure_ascii=False)
            json_file.write("\n")  # 写入换行符以分隔每个JSON对象


all_syn_label = []
with open('./data_preprocess/syndrome_knowledge.json','r',encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        syn_label = data['Name']
        all_syn_label.append(syn_label)

label = list(set(label))
num = 0
for l in label:
    if l in all_syn_label:
        num += 1

print(len(label))
print(num)