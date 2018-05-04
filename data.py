import os
import sys

import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from gensim.models.word2vec import Word2Vec

import jieba

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


if __name__ == '__main__':

    use_stop = True
    train_data_filepath = "./train_data.csv"
    test_data_filepath = "./test_data.csv"
    stop_list_path = "stop.txt"
    word_embedding_size = 200
    word2vec_model_path = "word2vec.model"

    # 读取数据
    train = pd.read_csv(train_data_filepath)
    test = pd.read_csv(test_data_filepath, encoding="gbk")

    titles_train = train["title"].tolist()
    titles_test = test["title"].tolist()

    titles = titles_train + titles_test

    stop_list = get_stop_list(stop_list_path)


    # 分词

    cut_titles = []

    for title in tqdm(titles):
        cut_title = jieba.cut(title)
        cut_title = [word for word in cut_title if use_stop and (word not in stop_list)]
        cut_titles.append(cut_title)

    print(len(cut_titles))
    print(cut_titles[0])

    # 训练词向量
    model = Word2Vec(cut_titles, size=word_embedding_size, window=5, min_count=5)
    model.save(word2vec_model_path)

    print(model.wv["股票"])

