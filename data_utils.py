from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
import datetime
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import pre_data
from tqdm import tqdm
import os
import json
from transformers import BertTokenizer
from sklearn.metrics import classification_report

model_path = './bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)
from transformers import BertConfig, BertModel, AdamW

model_config = BertConfig.from_pretrained(model_path)
label_model = BertModel.from_pretrained(model_path, config=model_config).to('cpu')

# 根据预测结果和标签数据来计算准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # report = classification_report(labels_flat,pred_flat)
    # 获取特定指标的值
    # print(report)
    # print(type(report))
    # accuracy = report['accuracy']
    # macro_f1 = report['macro avg']
    # macro_recall = report['macro avg']
    # macro_precision = report['macro avg']
    #
    # print(macro_f1)
    # print(macro_recall)
    # print(macro_precision)

    acc = np.sum(pred_flat == labels_flat) / len(labels_flat)
    precision = precision_score(labels_flat, pred_flat, average='macro', zero_division=1)
    recall = recall_score(labels_flat, pred_flat, average='macro', zero_division=1)
    f1 = f1_score(labels_flat, pred_flat, average='macro', zero_division=1)

    # return accuracy,macro_precision,macro_recall,macro_f1
    return acc,precision,recall,f1


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def Data_Loader(args,tokenizer):
    syndromes = []
    with open('./data_preprocess/syndrome_vocab.txt', 'r', encoding='utf-8') as file:
        for line in file:
            syndromes.append(line.replace('\n', ''))
    syndromes_text = {}
    with open('./data_preprocess/syndrome_knowledge.json', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            name = data['Name']
            text = data['Definition'] + '[SEP]' + data['Typical_performance'] + '[SEP]' + data['Common_isease']
            syndromes_text[name] = text

    # syndromes = tools.get_syndromes('./TCM/data_preprocess/syndrome_vocab.txt')
    id2syndrome_dict = {}
    syndrome2id_dict = {}
    id2syndrome_text = {}

    match_labels = []
    input_ids_text = []
    attention_masks_text = []
    num_lable = len(syndromes)
    num = []
    y = 0
    for i in tqdm(range(len(syndromes))):
        id2syndrome_dict[i] = syndromes[i]
        syndrome2id_dict[syndromes[i]] = i
        if syndromes[i] in syndromes_text:
            id2syndrome_text[i] = syndromes_text[syndromes[i]]
        else:
            id2syndrome_text[i] = '[pad]'

        if len(os.listdir('./def_vec/')) == num_lable:
            # 用于存储加载的张量的列表
            all_tensors = []
            # 逐个加载文件中的张量并添加到列表中
            for file_path in range(num_lable):
                loaded_tensor = torch.load('./def_vec/label_feature{}.pt'.format(file_path))
                all_tensors.append(loaded_tensor)
            label_feature = torch.stack(all_tensors, dim=0).to(args.device).view(-1, 512, 1024)
        else:
            match_text = syndromes_text[syndromes[i]].replace("\n", " ").replace("!", "\t").replace("?", "\t").replace(
                ".",
                "\t").replace(".",
                              "\t").replace(
                ",", "\t").replace("/", "\t")
            num.append(len(match_text))
            match_text = match_text[:510]
            encoded_dict_text = tokenizer.encode_plus(
                match_text,  # 输入文本
                add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                max_length=512,  # 填充 & 截断长度
                padding='max_length',
                return_attention_mask=True,  # 返回 attn. masks.
                return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                truncation=True
            )
            input_id_text = encoded_dict_text['input_ids'][0].to('cpu')
            attention_mask_text = encoded_dict_text['attention_mask'][0].to('cpu')
            label_fea = label_model(input_id_text.view(-1, 512),
                                    token_type_ids=None,
                                    attention_mask=attention_mask_text.view(-1, 512),
                                    return_dict=False)
            x = label_fea[0].to(args.device)
            torch.save(x, './def_vec/label_feature{}.pt'.format(y))
            y = y + 1

    # 输入文件路径，输出Tensor变量
    def get_InputTensor(path):
        contents = pre_data.read_json(path)
        sentences = []
        labels = []
        input_ids = []
        attention_masks = []
        true_splitNumbers = []


        # for content in tqdm(contents):
        #     sentence = content['chief_complaint'] + '[SEP]' + content['description'] + '[SEP]' + content['detection']
        #     sentence = sentence.replace("\n", " ").replace("!", "\t").replace("?", "\t").replace(".", "\t").replace(".",
        #                                                                                                             "\t").replace(
        #         ",", "\t").replace("/", "\t")
        #     LenSentence = len(sentence)
        #
        #     # #更改原始
        #     # LenSentence = 510
        #
        #     if LenSentence <= 510:
        #         sentence1 = sentence[0:LenSentence]
        #     else:
        #         sentence1 = sentence[0:510]
        #         sentence2 = sentence[510:LenSentence]
        #         true_splitNumber = 2
        #     sentences.append(sentence1)
        #     encoded_dict1 = tokenizer.encode_plus(
        #         sentence1,  # 输入文本
        #         add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
        #         max_length=512,  # 填充 & 截断长度
        #         padding='max_length',
        #         return_attention_mask=True,  # 返回 attn. masks.
        #         return_tensors='pt',  # 返回 pytorch tensors 格式的数据
        #         truncation=True
        #     )
        #     input_id = encoded_dict1['input_ids'][0]
        #     attention_mask = encoded_dict1['attention_mask'][0]
        #     input_ids.append(torch.unsqueeze(input_id, dim=0))
        #     attention_masks.append(torch.unsqueeze(attention_mask, dim=0))
        #     labels.append(syndrome2id_dict[content['norm_syndrome']])
        #
        # input_ids = torch.stack(input_ids, dim=0)
        # attention_masks = torch.stack(attention_masks, dim=0)
        # labels = torch.tensor(labels)

        for content in contents:
            sentence = content['description'] + '[SEP]' + content['detection']
            sentence = sentence.replace("\n", " ").replace("!", "\t").replace("?", "\t").replace(".", "\t").replace(".",
                                                                                                                    "\t").replace(
                ",", "\t").replace("/", "\t")
            sentences.append(sentence)
            labels.append(syndrome2id_dict[content['norm_syndrome']])
        for sentence in tqdm(sentences):
            LenSentence = len(sentence)
            sentence1 = ''
            sentence2 = ''

            true_splitNumber = 0
            if LenSentence <= 510:
                sentence1 = sentence[0:LenSentence]
                true_splitNumber = 1
            else:
                sentence1 = sentence[0:510]
                sentence2 = sentence[510:LenSentence]
                true_splitNumber = 2

            input_ids1 = torch.zeros(512, dtype=torch.int64)
            input_ids2 = torch.zeros(512, dtype=torch.int64)

            attention_mask1 = torch.zeros(512, dtype=torch.int64)
            attention_mask2 = torch.zeros(512, dtype=torch.int64)

            if true_splitNumber >= 1:
                encoded_dict1 = tokenizer.encode_plus(
                    sentence1,  # 输入文本
                    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                    max_length=512,  # 填充 & 截断长度
                    padding='max_length',
                    return_attention_mask=True,  # 返回 attn. masks.
                    return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                    truncation=True
                )
                input_ids1 = encoded_dict1['input_ids'][0]
                attention_mask1 = encoded_dict1['attention_mask'][0]
            if true_splitNumber >= 2:
                encoded_dict2 = tokenizer.encode_plus(
                    sentence2,  # 输入文本
                    add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                    max_length=512,  # 填充 & 截断长度
                    padding='max_length',
                    return_attention_mask=True,  # 返回 attn. masks.
                    return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                    truncation=True
                )
                input_ids2 = encoded_dict2['input_ids'][0]
                attention_mask2 = encoded_dict2['attention_mask'][0]

                # 将编码后的文本加入到列表

            input_ids.append(torch.stack([input_ids1, input_ids2], dim=0))
            attention_masks.append(torch.stack([attention_mask1, attention_mask2], dim=0))
            true_splitNumbers.append(true_splitNumber)
            # print(len(input_ids[0][0]))
        input_ids = torch.stack(input_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        labels = torch.tensor(labels)
        true_splitNumbers = torch.tensor(true_splitNumbers)

        return input_ids, attention_masks, labels, label_feature, true_splitNumbers

    # 输入文件路径，输出Tensor变量
    def get_InputTensor_chief(path):
        contents = pre_data.read_json(path)
        sentences_chief = []
        input_ids_chief = []
        attention_masks_chief = []
        true_splitNumbers_chief = []


        for content in contents:
            sentence_chief = content['chief_complaint']
            sentence_chief = sentence_chief.replace("\n", " ").replace("!", "\t").replace("?", "\t").replace(".", "\t").replace(
                ".",
                "\t").replace(
                ",", "\t").replace("/", "\t")
            sentences_chief.append(sentence_chief)
        for sentence_chief in tqdm(sentences_chief):
            LenSentence = len(sentence_chief)
            sentence1 = ''
            sentence2 = ''

            true_splitNumber = 0
            if LenSentence <= 510:
                sentence1 = sentence_chief[0:LenSentence]
                true_splitNumber = 1
            else:
                sentence1 = sentence_chief[0:510]
                sentence2 = sentence_chief[510:LenSentence]
                true_splitNumber = 2

            input_ids1 = torch.zeros(512, dtype=torch.int64)


            attention_mask1 = torch.zeros(512, dtype=torch.int64)



            encoded_dict1 = tokenizer.encode_plus(
                sentence1,  # 输入文本
                add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                max_length=512,  # 填充 & 截断长度
                padding='max_length',
                return_attention_mask=True,  # 返回 attn. masks.
                return_tensors='pt',  # 返回 pytorch tensors 格式的数据
                truncation=True
            )
            input_ids1 = encoded_dict1['input_ids'][0]
            attention_mask1 = encoded_dict1['attention_mask'][0]

            input_ids_chief.append(input_ids1)
            attention_masks_chief.append(attention_mask1)
            true_splitNumbers_chief.append(true_splitNumber)

        input_ids_chief = torch.stack(input_ids_chief, dim=0)
        attention_masks_chief = torch.stack(attention_masks_chief, dim=0)
        true_splitNumbers_chief = torch.tensor(true_splitNumbers_chief)

        return input_ids_chief, attention_masks_chief, true_splitNumbers_chief

    input_ids__train_chief, attention_masks_train_chief, true_splitNumbers_train_chief = get_InputTensor_chief(
        './data_preprocess/clear_train.json')
    input_ids_test_chief, attention_masks_test_chief, true_splitNumbers_test_chief = get_InputTensor_chief(
        './data_preprocess/clear_test.json')
    input_ids_val_chief, attention_masks_val_chief, true_splitNumbers_dev_chief = get_InputTensor_chief(
        './data_preprocess/clear_dev.json')

    input_ids_train, attention_masks_train, labels_train, label_feature,true_splitNumbers_train = get_InputTensor(
        './data_preprocess/clear_train.json')
    input_ids_test, attention_masks_test, labels_test, label_feature,true_splitNumbers_test = get_InputTensor(
        './data_preprocess/clear_test.json')
    input_ids_val, attention_masks_val, labels_val, label_feature,true_splitNumbers_dev = get_InputTensor('./data_preprocess/clear_dev.json')



    # 将输入数据合并为 TensorDataset 对象
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train,true_splitNumbers_train, input_ids__train_chief, attention_masks_train_chief, true_splitNumbers_train_chief)
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test,true_splitNumbers_test, input_ids_test_chief, attention_masks_test_chief, true_splitNumbers_test_chief)
    val_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val,true_splitNumbers_dev, input_ids_val_chief, attention_masks_val_chief, true_splitNumbers_dev_chief)

    # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  # 训练样本
        sampler=RandomSampler(train_dataset),  # 随机小批量
        batch_size=args.batch_size  # 以小批量进行训练
    )

    # 测试集不需要随机化，这里顺序读取就好
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,  # 验证样本
        sampler=SequentialSampler(test_dataset),  # 顺序选取小批量
        batch_size=args.batch_size
    )

    # 验证集不需要随机化，这里顺序读取就好
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,  # 验证样本
        sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
        batch_size=args.batch_size
    )

    return train_dataloader,test_dataloader,val_dataloader,syndromes_text,id2syndrome_dict,label_feature