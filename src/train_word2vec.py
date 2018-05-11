import os
import argparse
import logging

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from utils import read_data

"""
训练word2vec模型
"""

if __name__ == '__main__':

    logging.root.setLevel(logging.INFO)

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")

    logger = logging.getLogger()

    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_size", type=int, default=20, help="词向量的维度")
    parser.add_argument("--min_count", type=int, default=1, help="word2vec模型中词典词语出现的最小次数，小于这个次数的词语将被舍弃")
    parser.add_argument("--store_path", type=str, default="word2vec.model", help="模型保存的路径")

    parser.add_argument("--data_path", type=str, required=True, help="数据路径")
    parser.add_argument("--models_path", type=str, required=True, help="模型路径")

    args, _ = parser.parse_known_args()

    embedding_size = args.embedding_size
    min_count = args.min_count
    store_path = args.store_path


    train_ids, train_titles = read_data(os.path.join(args.data_path, "cut_train.csv"))
    test_ids, test_titles = read_data(os.path.join(args.data_path, "cut_test.csv"))

    
    logger.info("Start train word2vec model...")
    model = Word2Vec(train_titles, size=embedding_size, min_count=min_count, workers=4, window=3)
    #  model2 = KeyedVectors.load("./news12g_bdbk20g_nov90g_dim128/news12g_bdbk20g_nov90g_dim128.model")
    #  model.wv.syn0 = model2.syn0
    #  model.train(train_titles, total_examples=len(train_titles), epochs=model.iter)
    model.save(os.path.join(args.models_path, store_path))
    logger.info("End trainning.")
