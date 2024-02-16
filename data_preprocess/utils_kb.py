# -*- coding：utf-8 -*-

import tqdm
import json
import xlwt
import xlrd

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm

def tfidf_similarity(s1, s2):

    cv = TfidfVectorizer(ngram_range=[1,2])
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    if norm(vectors[0]) * norm(vectors[1]) == 0:
        return 0.0
    else:
        return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

def complete_kb(kb_indict, current_syndrome):
    acc = 0
    fault = []
    for k, v in current_syndrome.items():
        if k in kb_indict.keys():
            acc += 1
        else:
            fault.append(k)
    return acc, fault

def read_knowledge_base(pathtoknowledge):
    wb = xlrd.open_workbook(filename=pathtoknowledge)
    sheet_1 = wb.sheet_by_name("Sheet1")
    instances = []
    # all_description=[]
    for line_no in range(1,2140):
        Name = sheet_1.cell(line_no, 4).value
        Definition = sheet_1.cell(line_no, 5).value
        # Definition = sheet_1.cell(line_no, 5).value.replace(Name,'').replace("中医病证名","").replace("中医病名",'').replace("，。",'').replace("，，",'')
        # Definition = Name+"的定义是"+Definition
        Typical_performance = sheet_1.cell(line_no, 6).value
        Common_isease  =sheet_1.cell(line_no, 7).value

        instance = {
            "Name":Name,
            "Definition": Definition,
            "Typical_performance": Typical_performance,
            "Common_isease": Common_isease,
        }

        instances.append(instance)
    unique_instances = [dict(t) for t in {tuple(d.items()) for d in instances}]
    kb_in_dict={}
    for syndrome in unique_instances:
        kb_in_dict[syndrome['Name']]=syndrome
    return kb_in_dict


