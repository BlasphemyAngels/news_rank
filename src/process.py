import os
import sys
import logging
import argparse
import re

from tqdm import tqdm
import jieba
import pandas as pd

from utils import read_data
from utils import get_stop_list
from utils import store_cut_data
from utils import tag_time_and_num

"""
    将数据分词去停用词，将时间数字标记成'TIME','NUM'标签。
"""

if __name__ == '__main__':
    
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")
    logging.root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--use_stop", type=bool, default=True, help="是否使用停用词")
    parser.add_argument("--train_data", type=str, default="train_data.csv", help="训练数据")
    parser.add_argument("--test_data", type=str, default="test_data.csv", help="数据")
    parser.add_argument("--stop_path", type=str, default="stop.txt", help="停用词表的文件路径")
    parser.add_argument("--data_path", type=str, required=True, help="数据路径")

    args, _ = parser.parse_known_args()

    data_path = os.path.realpath(args.data_path)
    train_data_path = os.path.join(data_path, args.train_data)
    test_data_path = os.path.join(data_path, args.test_data)

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path, encoding="gbk")

    titles_train = train["title"].tolist()
    titles_test = test["title"].tolist()

    stop_list_path = os.path.join(data_path, args.stop_path)
    stop_list = get_stop_list(stop_list_path)

    titles_train = tag_time_and_num(titles_train)
    titles_test = tag_time_and_num(titles_test)

    cut_titles_train = []
    for title in tqdm(titles_train):
        cut_title_train = jieba.cut(title)
        cut_title_train = [word for word in cut_title_train if args.use_stop and (word not in stop_list)]
        cut_titles_train.append(cut_title_train)

    cut_titles_test = []

    for title in tqdm(titles_test):
        cut_title_test = jieba.cut(title)
        cut_title_test = [word for word in cut_title_test if args.use_stop and (word not in stop_list)]
        cut_titles_test.append(cut_title_test)

    store_cut_data(os.path.join(data_path, "cut_train.csv"), train, cut_titles_train)
    store_cut_data(os.path.join(data_path, "cut_test.csv"), test, cut_titles_test)

