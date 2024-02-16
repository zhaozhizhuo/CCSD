# -*- coding：utf-8 -*-
'''
Author: Mucheng Ren
Function: includes several functions that frequently imported in other scripts.
'''
import json
import xlrd
import random
from math import ceil
from utils_kb import read_knowledge_base,complete_kb
from os import path
from imageio import imread
import matplotlib.pyplot as plt
import os
import jieba
from wordcloud import WordCloud, ImageColorGenerator

class Syndrome():
    syncount=0
    syntype=[]
    unnormalized_syndrome = []
    normalizaed_syndrome=[]
    action = []
    normalization_map ={}
    with open('syndrome.txt', 'r') as f:
        for line in f:
            [u,n,a,_]= line.strip('\n').split(',')
            unnormalized_syndrome.append(u)
            normalizaed_syndrome.append(n)
            action.append(a)
            normalization_map[u] = n

    def __init__(self,name):
        self.name = name
        if name not in Syndrome.syntype:
            Syndrome.syntype.append(name)
            Syndrome.syncount +=1

    def displayCount(self):
        print("Total Syndrome types %d" % Syndrome.syncount)

    def normalize(self,name):
        new_name = Syndrome.normalization_map[name]
        self.name=new_name

class TCM_case():

    TCMcasecount = 0
    def __init__(self, syndrome, disease, chief, history, detection):
        self.syndrome = Syndrome(syndrome)
        self.disease = disease
        self.chief = chief
        self.history = history
        self.detection = detection
        TCM_case.TCMcasecount+=1

    def displayCount(self):
        print("Total valid cases %d" % TCM_case.TCMcasecount)


def build_syndrome_map(datapath ='syndrome.txt'):
    normalization_map ={}
    valid_syndrome_number = 0
    with open(datapath, 'r') as f:
        for line in f:
            [u,n,a,d]= line.strip('\n').split(',')
            normalization_map[u.strip()] = {
                "nor_syndrome":n.strip(),
                "action":a.strip(),
                "description":d,
            }
            if a.strip() =='KEEP':
                valid_syndrome_number+=1
    print("total valid syndrome %d" % valid_syndrome_number)
    return normalization_map

undefined_syndrome =[]
def normalize_syndrome(name,normalization_map,disease=None):
    global undefined_syndrome
    final_name =""
    if name in normalization_map.keys():
        action = normalization_map[name]['action']
        if action == 'KEEP':
            final_name = name
        if action == 'MERGE':
            final_name = normalization_map[name]['nor_syndrome']
            while normalization_map[final_name]['action'] == 'MERGE':
                final_name = normalization_map[final_name]['nor_syndrome']
        if action == 'DELETE':
            final_name = ''
        if action == 'SUB_MERGE':
            if disease == "崩漏病":
                final_name = "脾气虚证"
            elif disease == "带下病":
                final_name = "脾虚湿盛证"
            else:
                final_name = ''
            # print(name,final_name)
    else:
        if name not in undefined_syndrome:
            print("An undefined syndrome: %s " % name)
        undefined_syndrome.append(name)

    return final_name



def if_contain_chaos(keyword):
    try:
        keyword.encode("gbk")
    except UnicodeEncodeError:
        return True
    return False


def read_data(data_path,sheet_name):
    wb = xlrd.open_workbook(filename=data_path)
    sheet_1 = wb.sheet_by_name(sheet_name)
    instances = []
    # all_description=[]
    for line_no in range(1,61134):
        BLH = sheet_1.cell(line_no, 0).value
        lcd_id = sheet_1.cell(line_no, 1).value
        lcd_name = sheet_1.cell(line_no, 2).value
        syndrome  =sheet_1.cell(line_no, 3).value
        chief_complaint = sheet_1.cell(line_no, 4).value.strip()
        History_of_present_illness = sheet_1.cell(line_no, 5).value.strip()
        WWQ = sheet_1.cell(line_no, 6).value.strip()

        if History_of_present_illness == "":
            continue

        instance = {
            # "line_no": line_no + 1,
            "user_id":BLH,
            "lcd_id": lcd_id,
            "lcd_name": lcd_name,
            "syndrome": syndrome,
            "chief_complaint": chief_complaint,
            "description": History_of_present_illness,
            "detection": WWQ,
        }

        instances.append(instance)
        # if [syndrome,chief_complaint,History_of_present_illness,WWQ] not in all_description:
        # all_description.append([syndrome,chief_complaint,History_of_present_illness,WWQ])
        # else:
    return instances


def filter_no_syndrome(instances):
    valid_instances= []
    fault_instances =[]

    for i in instances:
        if i["norm_syndrome"]=="" or i["norm_syndrome"]=="NULL":
            fault_instances.append(i)
        else:
            valid_instances.append(i)
    return fault_instances,valid_instances


def filter_multi_syndrome(instances):
    multi_syndrome_instances = []
    single_syndrome_instances=[]
    for i in instances:
        if "、" in i["syndrome"]:
            multi_syndrome_instances.append(i)
        else:
            single_syndrome_instances.append(i)
    return multi_syndrome_instances,single_syndrome_instances


def filter_undersampled_syndrome(syndrome_dict,threshold=1):

    undersample_instances = {}
    valid_instances={}
    for syn,ins in syndrome_dict.items():
        if len(ins) <=threshold:
            print("%s should be deleted, has less than %d instances" % (syn,threshold))
            undersample_instances[syn] =ins
        else:
            valid_instances[syn] = ins
    return valid_instances,undersample_instances


def build_syndrome_type_dict(valid_ins,out_path='syndrome_vocab.txt'):
    syndrome_key_dict={}
    writer = open(out_path,'w+',encoding='utf-8')
    for key,val in valid_ins.items():
        writer.write(key)
        writer.write('\n')
        syndrome_key_dict[key] = len(val)
    writer.close()
    return syndrome_key_dict

def split_train_dev_syndrome(valid_ins,ratio,shuffle=False,out_train_path='train_v1.json',out_dev_path='dev_v1.json',out_test_path='test_v1.json'):
    train_split = []
    dev_split = []
    test_split=[]
    for syn,ins in valid_ins.items():
        dev_offset = ceil(len(ins)*ratio[1])
        test_offset = ceil(len(ins)*ratio[2])
        if shuffle:
            random.shuffle(ins)
        dev_split.extend(ins[:dev_offset])
        test_split.extend(ins[dev_offset:dev_offset+test_offset])
        train_split.extend(ins[dev_offset+test_offset:])

    train_writer = open(out_train_path,'w+',encoding='utf-8')
    dev_writer = open(out_dev_path,'w+',encoding='utf-8')
    test_writer = open(out_test_path,'w+',encoding='utf-8')


    for t in train_split:
        train_writer.write(json.dumps(t,ensure_ascii=False))
        train_writer.write('\n')
    for d in dev_split:
        dev_writer.write(json.dumps(d,ensure_ascii=False))
        dev_writer.write('\n')
    for te in test_split:
        test_writer.write(json.dumps(te,ensure_ascii=False))
        test_writer.write('\n')
    train_writer.close()
    dev_writer.close()
    test_writer.close()

    print("Train set has %d instances, Dev set has %d instances,Test set has %d instances" %(len(train_split),len(dev_split),len(test_split)))
    return train_split,dev_split,test_split


def compute_vocab_statistics(data):
    word_count = []

    for d in data:
        for i in d["chief_complaint"]:
            word_count.extend(i)
        for i in d['description']:
            word_count.extend(i)
        for i in d['detection']:
            word_count.extend(i)
        for i in d['norm_syndrome']:
            word_count.extend(i)
    word_count_final = set(word_count)
    return word_count_final



def make_word_cloud(text):

    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    stopwords_path = d + '/wc_cn/stopwords_cn_en.txt'
    font_path = r'C:\Windows\Fonts\STKAITI.TTF'

    # The function for processing text with Jieba
    def jieba_processing_txt(text):

        mywordlist = []
        seg_list = jieba.cut(text, cut_all=False)
        liststr = "/ ".join(seg_list)

        with open(stopwords_path, encoding='utf-8') as f_stop:
            f_stop_text = f_stop.read()
            f_stop_seg_list = f_stop_text.splitlines()
            f_stop_seg_list.extend(["入院","进一步","诊治","我院","就诊","门诊","求治","治疗","收住","我科","患者","本院"])

        for myword in liststr.split('/'):
            if not (myword.strip() in f_stop_seg_list) and len(myword.strip()) > 1:
                mywordlist.append(myword)
        return ' '.join(mywordlist)


    wc = WordCloud(font_path=font_path, background_color="white",scale=64,max_font_size=40,max_words=100)


    wc.generate(jieba_processing_txt(text))


    # plt.figure()
    plt.rcParams['figure.dpi'] = 500
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    # save wordcloud
    wc.to_file(path.join(d, "wc_full_MH.jpg"))


if __name__ == "__main__":
    instance= read_data("data.xlsx",sheet_name="all")
    map = build_syndrome_map()
    for i in instance:
        i['norm_syndrome'] = normalize_syndrome(i['syndrome'],map,i['lcd_name'])

    f,v = filter_no_syndrome(instance)


    syndrome_dict = {}
    for i in v:
        if i['norm_syndrome'] not in syndrome_dict.keys():
            syndrome_dict[i['norm_syndrome']] = [i]
        else:
            syndrome_dict[i['norm_syndrome']].append(i)


    valid_Ins, unders_Ins = filter_undersampled_syndrome(syndrome_dict,threshold=10)
    # print(len(valid_Ins))

    syndrome_statistics = build_syndrome_type_dict(valid_Ins)

    duplicate_syndrome = [i for i,j in syndrome_statistics.items()]

    # dict_kb = read_knowledge_base()
    # a,b = complete_kb(dict_kb,syndrome_statistics)


    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    syndrome_statistics = sorted(syndrome_statistics.items(), key=lambda x: x[1], reverse=True)
    shapes = [i[0] for i in syndrome_statistics[:19] ]
    shapes_en = ["syndrome of qi deficiency with blood stasis",
                 "syndrome of dampness and heat pouring downwards",
                 "syndrome of qi stagnation and blood stasis",
                 "syndrome of blockade of wind-phlegm-static blood",
                 "syndrome of stagnant heat in the liver and stomach",
                 "syndrome of stagnated healthy qi and pathogenic factors",
                 "syndrome of phlegm and dampness accumulating in the lung",
                 "syndrome of liver and kidney depletion",
                 "syndrome of heat toxin congestion and binding",
                 "syndrome of dual deficiency of spleen and kidney",
                 "syndrome of qi and yin deficiency",
                 "syndrome of dampness and heat stagnation and obstruction",
                 "syndrome of kidney deficiency",
                 "syndrome of disharmony of liver and stomach",
                 "syndrome of wind and cold attacking the lung",
                 "syndrome of yang deficiency in  spleen and stomach",
                 "syndrome of dampness and heat accumulation and binding",
                 "syndrome of yang deficiency with water flooding",
                 "syndrome of impediment and obstruction of phlegm and stasis",
                 "others"
                 ]
    shapes_en = ["S.QDBS",
                 "S.DHPD",
                 "S.QSBS",
                 "S.BWPSB",
                 "S.SHLS",
                 "S.SHQPF",
                 "S.PDAL",
                 "S.LKP",
                 "S.HTCB",
                 "S.DDSK",
                 "S.QYD",
                 "S.DHSO",
                 "S.KD",
                 "S.DLS",
                 "S.WCAL",
                 "S.YDSS",
                 "S.DHAB",
                 "S.YDWF",
                 "S.IBPS",
                 "Others"
                 ]
    values = [i[1] for i in syndrome_statistics[:19] ]
    shapes.append("其他")
    values.append(sum([i[1] for i in syndrome_statistics[19:]]))
    plt.rcParams['figure.dpi'] = 400
    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=shapes_en, autopct='%1.1f%%',
            shadow=False, startangle=270)
    ax1.axis('equal')
    plt.savefig('syndrome_analysis_en.jpg')
    plt.show()




    train_split,dev_split,test_split = split_train_dev_syndrome(valid_Ins,ratio=[0.8,0.1,0.1],shuffle=False,out_train_path='train.json',out_dev_path='dev.json',out_test_path='test.json')
    #

