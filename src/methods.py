import os
import sys

import heapq

from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim import models
from gensim import corpora
from gensim import similarities

from tqdm import tqdm

from utils import read_data
from utils import filter_words
from utils import write_res
from utils import get_vecs_by_pca_and_w2v
from utils import get_vecs
from utils import n_largest_similarity
from utils import lcs_distance
from utils import jaccard

def doc2vec(args):

    model = Doc2Vec.load(args["model_path"])

    test_ids, test_titles = read_data(args["infer_data"])
    train_ids, _ = read_data(args["train_data"])

    l_data = len(test_ids)

    res = []

    for i in tqdm(range(l_data)):
        source_id = test_ids[i]
        source_title = test_titles[i]

        source_title_vec = model.infer_vector(source_title)

        n_sims = model.docvecs.most_similar([source_title_vec], topn=25)

        sim_ids = []
        for sim in n_sims:
            sim_ids.append(train_ids[int(sim[0])])

        #  while(str(source_id) in sim_ids):
            #  sim_ids.remove(str(source_id))

        for j in range(20):
            res.append([source_id, sim_ids[j]])

    write_res("res/doc2vec.txt", res)

def tfidf_word2vec(args):

    train_data = args["train_data"]
    test_data = args["infer_data"]
    model_path = args["model_path"]

    train_ids, train_titles = read_data(train_data)
    test_ids, test_titles = read_data(test_data)

    dictionary = corpora.Dictionary(train_titles)

    corpus = [dictionary.doc2bow(text) for text in train_titles]

    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    model = Word2Vec.load(model_path)
    embedding_size = len(model.wv[train_titles[0][0]])

    train_vecs = get_vecs(train_titles, dictionary, tfidf_model, embedding_size, model)
    test_vecs = get_vecs(train_titles, dictionary, tfidf_model, embedding_size, model)

    n_best_sims = n_largest_similarity(20, train_ids, train_vecs, test_ids, test_vecs)

    write_res("res/res_tfidf_w2v.txt", n_best_sims)

def lcs(args):

    train_data = args["train_data"]
    test_data = args["infer_data"]

    train_ids, train_titles = read_data(train_data)
    test_ids, test_titles = read_data(test_data)

    l_train = len(train_ids)
    l_test = len(test_ids)

    res = []
    for i in tqdm(range(l_test)):
        source_id = test_ids[i]

        sims = []

        for j in range(l_train):
            sims.append((train_ids[j], lcs_distance(test_titles[i], train_titles[j])))

        n_largest_sims = heapq.nlargest(21, sims, key=lambda x: x[1])

        n_largest_sims_ids = list(map(lambda x: x[0], n_largest_sims))

        if(source_id in n_largest_sims_ids):
            n_largest_sims_ids.remove(source_id)

        n_largest_sims_ids = n_largest_sims_ids[0:20]

        for target_id in n_largest_sims_ids:
            res.append((source_id, target_id))

    write_res("res/res_lcs.txt", res)

def jaccardSim(args):

    train_data = args["train_data"]
    test_data = args["infer_data"]

    train_ids, train_titles = read_data(train_data)
    test_ids, test_titles = read_data(test_data)

    l_train = len(train_titles)
    l_test = len(test_titles)

    res = []
    for i in tqdm(range(l_test)):
        source_id = test_ids[i]

        source_set = set(test_titles[i])

        sims = []
        for j in range(l_train):
            target_id = train_ids[j]

            target_set = set(train_titles[j])

            similarity = jaccard(source_set, target_set)

            sims.append((target_id, similarity))


        n_largest_sims = heapq.nlargest(21, sims, lambda x: x[1])
        n_largest_sims_ids = [x[0] for x in n_largest_sims if x[0] != source_id][0:20]
        
        for target_id in n_largest_sims_ids:
            res.append((source_id, target_id))


    write_res("res/jacarrd.txt", res)

def tfidf(args):
    train_ids, train_titles = read_data(args["train_data"])
    test_ids, test_titles = read_data(args["infer_data"])

    dictionary = corpora.Dictionary(train_titles)
    corpus = [dictionary.doc2bow(text) for text in train_titles]

    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    simility = similarities.Similarity("Similarity-tfidf-index", corpus_tfidf, num_features=len(dictionary))

    res = []

    simility.num_best = 21
    for i in range(len(test_ids)):
        id_ = test_ids[i]
        title_ = test_titles[i]

        title_bow = dictionary.doc2bow(title_)

        title_bow_tfidf = tfidf_model[title_bow]
        
        #  title_bow_lsi = lsi[title_bow_tfidf]

        best_sims = simility[title_bow_tfidf]
        #  best_sims.reverse()

        for best_sim in best_sims:

            if(id_ != train_ids[best_sim[0]]):
                res.append([id_, train_ids[best_sim[0]]])

    write_res("res/tfidf.txt", res)
