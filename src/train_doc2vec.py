import os
import logging
import argparse

from utils import read_data

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

if __name__ == '__main__':

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s")
    logging.root.setLevel(logging.INFO)
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default="cut_train.csv", help="训练数据")
    parser.add_argument("--embedding_size", type=int, default=128, help="嵌入向量维度")
    parser.add_argument("--window", type=int, default=5, help="模型训练滑动窗口大小")
    parser.add_argument("--min_count", type=int, default=5, help="词语的最小频度")
    parser.add_argument("--store_path", type=str, default="doc2vec.model", help="模型保存的路径")
    parser.add_argument("--data_path", type=str, required=True, help="数据路径")
    parser.add_argument("--models_path", type=str, required=True, help="模型存储路径")
    args, _ = parser.parse_known_args()

    data_path = os.path.realpath(args.data_path)

    train_ids, train_titles = read_data(os.path.join(data_path, args.train_data))

    documents = []
    L_data = len(train_ids)
    for i in tqdm(range(L_data)):
        document = TaggedDocument(train_titles[i], tags=[train_ids[i]])
        documents.append(document)

    model = Doc2Vec(documents, size=args.embedding_size, window=args.window, min_count=args.min_count)
    model.save(os.path.join(os.path.realpath(args.models_path), args.store_path))
