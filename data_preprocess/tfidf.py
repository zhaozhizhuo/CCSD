# -*- coding：utf-8 -*-

from gensim import corpora, models, similarities
import numpy as np
from utils_kb import read_knowledge_base
import json
import random
import jieba
import tqdm
def similarity(query_path, query):
    """
    :func: 计算问题与知识库中问题的相似度
    :param query_path: 问题文件所在路径
    :param query: 所提问题
    :return: 返回满足阈值要求的问题所在行索引——对应答案所在的行索引

    """
    class MyCorpus():
        def __iter__(self):
            for line in open(query_path, 'r', encoding='utf-8'):
                 yield line.split()

    Corp = MyCorpus()
    # 建立词典
    dictionary = corpora.Dictionary(Corp)

    # 基于词典，将分词列表集转换成稀疏向量集，即语料库
    corpus = [dictionary.doc2bow(text) for text in Corp]

    # 训练TF-IDF模型，传入语料库进行训练
    tfidf = models.TfidfModel(corpus)

    # 用训练好的TF-IDF模型处理被检索文本，即语料库
    corpus_tfidf = tfidf[corpus]

    # # 得到TF-IDF值
    # for temp in corpus_tfidf:
    #     print(temp)

    vec_bow = dictionary.doc2bow(query.split())
    vec_tfidf = tfidf[vec_bow]

    index = similarities.MatrixSimilarity(corpus_tfidf)
    sims = index[vec_tfidf]
    max_loc = np.argmax(sims)
    max_sim = sims[max_loc]
    # 句子相似度阈值
    sup = 0.7
    # row_index默认为-1，即未匹配到满足相似度阈值的问题
    row_index = -1
    if max_sim > sup:
        # 相似度最大值对应文件中问题所在的行索引
        row_index = max_loc + 1

    return row_index


if __name__ == '__main__':
    kb = read_knowledge_base()
    labels = []
    with open("syndrome_vocab.txt", 'r', encoding='utf-8') as f:
        for line in f:
            labels.append(line.strip('\n'))
    with open("path\to\data\XXX.json", 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    writer =open("path", 'w+', encoding='utf-8')
    all_knowledge_definition = [kb[l]['Definition'] for l in labels if l in kb.keys()]
    all_knowledge_definition_syndrome = [l for l in labels if l in kb.keys()]
    Corp = [list(jieba.cut(i)) for i in all_knowledge_definition]
    dictionary = corpora.Dictionary(Corp)
    corpus = [dictionary.doc2bow(text) for text in Corp]
    tfidf = models.TfidfModel(corpus)
    # corpus_tfidf = tfidf[corpus]
    #

    # # for temp in corpus_tfidf:
    # #     print(temp)
    #
    # predicts =[]
    # reference=[]
    # top_predicts = []
    # Train = True
    #
    # for line in tqdm.tqdm(lines):
    #     # initialize current valid knowledge
    #     data_raw = json.loads(line.strip('\n'))
    #     user_id = data_raw['user_id']
    #     lcd_id = data_raw['lcd_id']
    #     lcd_name = data_raw['lcd_name']
    #     syndrome = data_raw['syndrome']
    #     chief_complaint = data_raw['chief_complaint']
    #     description = data_raw['description']
    #     detection = data_raw['detection']
    #     norm_syndrome = data_raw['norm_syndrome']
    #
    #     # vec_bow = dictionary.doc2bow(list(jieba.cut(description)))
    #     # vec_tfidf = tfidf[vec_bow]
    #     # index = similarities.MatrixSimilarity(corpus_tfidf)
    #     # sims = index[vec_tfidf]
    #     # max_loc = np.argmax(sims)
    #     reference.append(labels.index(norm_syndrome))
    #     # predicts.append(max_loc)
    #     # top_predicts.append([i for i in np.argpartition(sims, -20)[-20:]])
    #     # top3_predicts = [i for i in np.argpartition(sims, -5)[-5:]]
    #
    # #     p1 = 0
    # #     p3 = 0
    # #     p5 = 0
    # #     p10 = 0
    # #     p20 = 0
    # #     for p in top_predicts:
    # #         if labels.index(norm_syndrome) in p[-1:]:
    # #             p1+=1
    # #         if labels.index(norm_syndrome) in p[-3:]:
    # #             p3+=1
    # #         if labels.index(norm_syndrome) in p[-5:]:
    # #             p5+=1
    # #         if labels.index(norm_syndrome) in p[-10:]:
    # #             p10+=1
    # #         if labels.index(norm_syndrome) in p[-20:]:
    # #             p20+=1
    # #
    # # print(p1/len(predicts))
    # # print(p3 / len(predicts))
    # # print(p5 / len(predicts))
    # # print(p10 / len(predicts))
    # # print(p20 / len(predicts))
    # #     if Train:
    # #         best_k = all_knowledge_definition[all_knowledge_definition_syndrome.index(norm_syndrome)]
    # #         if labels.index(norm_syndrome) in top3_predicts:
    # #             top3_predicts.remove(labels.index(norm_syndrome))
    # #         other_k = [all_knowledge_definition[i] for i in top3_predicts]
    # #         full_k = [{norm_syndrome:best_k},{labels[top3_predicts[0]]:other_k[0]},{labels[top3_predicts[1]]:other_k[1]},{labels[top3_predicts[2]]:other_k[2]}]
    # #         # print(full_k)
    # #         random.shuffle(full_k)
    # #         knowledge_para = ' '.join([list(i.values())[0] for i in full_k])
    # #         data_raw['knowledge_option'] = [list(i.keys())[0] for i in full_k]
    # #     else:
    # #         full_k = [all_knowledge_definition[i] for i in np.argpartition(sims, -3)[-3:]]
    # #         knowledge_para = ' '.join(full_k)
    # #     data_raw["knowledge_para"]= knowledge_para
    # #
    # #     writer.write(json.dumps(data_raw,ensure_ascii=False))
    # #     writer.write('\n')
    #
    #     # For random
    #     best_k = all_knowledge_definition[all_knowledge_definition_syndrome.index(norm_syndrome)]
    #     duplicate_k= [ i for i in all_knowledge_definition_syndrome]
    #     duplicate_k.remove(norm_syndrome)
    #     random_k = random.sample(duplicate_k,4)
    #     random_k_definition = [all_knowledge_definition[all_knowledge_definition_syndrome.index(i)] for i in random_k]
    #     full_k = [{norm_syndrome: best_k}, {random_k[0]: random_k_definition[0]}, {random_k[1]: random_k_definition[1]},{random_k[2]: random_k_definition[2]},{random_k[3]: random_k_definition[3]}]
    #     random.shuffle(full_k)
    #     knowledge_para = ' '.join([list(i.values())[0] for i in full_k])
    #     data_raw['knowledge_option'] = [list(i.keys())[0] for i in full_k]
    #     data_raw["knowledge_para"]= knowledge_para
    #     writer.write(json.dumps(data_raw,ensure_ascii=False))
    #     writer.write('\n')
    # writer.close()

