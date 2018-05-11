import os
import sys
import re
from tqdm import tqdm
import numpy as np
import heapq

from sklearn.decomposition import PCA
def write_res(filename, data):

    w_data = list(map(lambda x: str(x[0]) + "\t" + str(x[1]) + "\n", data))
    with open(filename, "w") as f:
        f.write("source_id" + "\t" + "target_id" + "\n")

        for d in w_data:
            f.write(d)

def read_data(filename):
    ids = []
    titles = []
    with open(filename, "r") as f:
        lines = f.readlines()
        
        for line in lines:
            lineSplit = line.split(",")
            ids.append(lineSplit[0])
            titles.append(lineSplit[1].split())
    return ids, titles

def filter_words(data, dictionary):

    f_data = []

    for d in data:

        ans = []

        for word in d:
            if(word in dictionary):
                ans.append(word)
        f_data.append(ans)
    return f_data

def get_vecs_by_pca_and_w2v(titles, vectors, dim=300):

    pca = PCA(dim)
    vecs = []
    for title in tqdm(titles):

        vec = []

        for word in title:
            vec.extend(vectors[word])

        vecs.append(vec)

    return pca.fit_transform(vecs)

def get_vecs(titles, dictionary, tfidf_model, embedding_size, model):
    vecs = []

    for title in titles:

        vec = [0 for i in range(embedding_size)]

        title_bow = dictionary.doc2bow(title)

        title_bow_tfidf = tfidf_model[title_bow]

        tfidf_sum = 0.0

        for word_tfidf in title_bow_tfidf:

            word = dictionary[word_tfidf[0]]
            tfidf_value = word_tfidf[1]

            word_vec = model.wv[word]

            tfidf_sum += tfidf_value

            for i in range(embedding_size):
                vec[i] += tfidf_value * word_vec[i]

        for i in range(embedding_size):
            vec[i] /= tfidf_sum
        vecs.append(vec)
    return vecs

def n_largest_similarity(n, train_ids, train_vecs, test_ids, test_vecs):

    l_train = len(train_ids)
    l_test = len(test_ids)

    print("Compute %d largest similarity.." % n)
    res = []

    for i in tqdm(range(l_test)):

        source_id = test_ids[i]

        source_vec = test_vecs[i]

        sims = []
        for j in range(l_train):

            target_id = train_ids[j]
            target_vec = train_vecs[j]

            similarity = eucl_sim(source_vec, target_vec)

            sims.append((target_id, similarity))

        n_largest_sims = heapq.nlargest(n + 1, sims, key=lambda x: x[1])
        
        n_largest_sims_ids = list(map(lambda x: x[0], n_largest_sims))
        
        if(source_id in n_largest_sims_ids):
            n_largest_sims_ids.remove(source_id)

        n_largest_sims_ids = n_largest_sims_ids[0:n]

        for id_ in n_largest_sims_ids:
            res.append((source_id, id_))

    return res

def get_similarity_by_vec(v1, v2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(v1, v2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    return dot_product / ((normA * normB) ** 0.5)

def eucl_sim(a,b):
    a = np.array(a)
    b = np.array(b)
    return 1/(1+np.sqrt((np.sum(a-b)**2)))

def lcs_distance(str_a, str_b):
    lensum = float(len(str_a) + len(str_b))
    #得到一个二维的数组，类似用dp[lena+1][lenb+1],并且初始化为0
    lengths = [[0 for j in range(len(str_b)+1)] for i in range(len(str_a)+1)]

    #enumerate(a)函数： 得到下标i和a[i]
    for i, x in enumerate(str_a):
        for j, y in enumerate(str_b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

    #到这里已经得到最长的子序列的长度，下面从这个矩阵中就是得到最长子序列
    result = ""
    x, y = len(str_a), len(str_b)
    while x != 0 and y != 0:
        #证明最后一个字符肯定没有用到
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else: #用到的从后向前的当前一个字符
            assert str_a[x-1] == str_b[y-1] #后面语句为真，类似于if(a[x-1]==b[y-1]),执行后条件下的语句
            result = str_a[x-1] + result #注意这一句，这是一个从后向前的过程
            x -= 1
            y -= 1
            
            #和上面的代码类似
            #if str_a[x-1] == str_b[y-1]:
            #    result = str_a[x-1] + result #注意这一句，这是一个从后向前的过程
            #    x -= 1
            #    y -= 1
    longestdist = lengths[len(str_a)][len(str_b)]
    ratio = longestdist/min(len(str_a),len(str_b))
    #return {'longestdistance':longestdist, 'ratio':ratio, 'result':result}
    return ratio

def jaccard(seta, setb):
    sa_sb = 1.0 * len(seta & setb) / len(seta | setb)
    return sa_sb

def get_stop_list(stop_list_name):
    """
    得到停用词表

    @param stop_list_name: 停用词表的文件名
    @return: 包含停用词表的list
    """
    
    with open(stop_list_name, "r") as f:
        stop_list = f.readlines()

    stop_list = list(map(lambda c: c.replace("\n", ""), stop_list))
    return stop_list

def store_cut_data(store_path, data, cut_titles):
    with open(store_path, "w") as f:
        ids = data["id"].tolist()
        L = len(ids)
        for i in range(L):
            id_ = ids[i]
            if(len(cut_titles[i]) == 0):
                continue
            title_ = " ".join(cut_titles[i])

            f.write(str(id_) + "," + title_ + "\n")

def filter_time_num_etc(a):

    t1 = "[\d]+年"
    t2 = "[\d]+年[\d]+月"
    t3 = "[\d]+年[\d]+月[\d]+日"
    t4 = "[\d]+月[\d]+日"

    
    tt = "[\d]+:[\d]+"

    tt1 = t1 + tt
    tt2 = t2 + tt
    tt3 = t3 + tt
    tt4 = t4 + tt

    a = re.sub(tt3, "TIME", a)
    a = re.sub(tt4, "TIME", a)
    a = re.sub(tt2, "TIME", a)
    a = re.sub(tt1, "TIME", a)
    a = re.sub(t3, "TIME", a)
    a = re.sub(t4, "TIME", a)
    a = re.sub(t2, "TIME", a)
    a = re.sub(t1, "TIME", a)
    a = re.sub(tt, "TIME", a)

    n1 = r"[\d]+\.?[\d]*%?"

    a = re.sub(n1, "NUM", a)
    return a

def tag_time_and_num(titles):
    """
    将文本中的时间和数字标记成TIME、NUM标记\
    @param titles: 要标记的文本序列，一个list，里面的每一个元素是一个字符串
    @return : list，每一个元素是标记完的字符串
    """
    new_titles = []
    for title in titles:
        new_titles.append(filter_time_num_etc(title))
    return new_titles
