import sys
from src.model_vMNVTM import vMNVTM

import argparse
from datasets import load_dataset, Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
import random
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

def train(gamma, temp, kappa1, kappa2, n_topics, n_epochs, lambda_sh):
    tm = vMNVTM(gamma = gamma, temp = temp, numb_embeddings=n_topics, embedding_dim=100, penalty=1,
                top_n_words=200, epochs=n_epochs, lambda_sh=lambda_sh,
                kappa1 = kappa1, kappa2 = kappa2)
    model_output = tm.train_model(df['train']['text'], df['test']['text'])

    return tm, model_output

def evaluate(tm, model_output, matric):
    if matric == 'keywords':
        topics = model_output.get('topics', [])
        for i, topic in enumerate(topics):
            print(f"Topic {i + 1}:")
            top_words = topic[:20]
            print(", ".join(top_words))
            print()

    if matric == 'diversity':
        score_diversity = TopicDiversity(topk=10).score(model_output)  # Compute score of the metric
        print("Diversity:", "{:.4f}".format(score_diversity))

    if matric == 'c_v':
        tm.tp.lemmas = [doc for doc in tm.tp.lemmas if len(doc) > 1]
        score_cv = Coherence(texts=tm.tp.lemmas, topk=10, measure='c_v').score(model_output)  # Compute score of the metric
        print("c_v:", "{:.4f}".format(score_cv))

    if matric == 'c_npmi':
        tm.tp.lemmas = [doc for doc in tm.tp.lemmas if len(doc) > 1]
        score_npmi = Coherence(texts=tm.tp.lemmas, topk=10, measure='c_npmi').score(model_output)  # Compute score of the metric
        print("c_npmi:", "{:.4f}".format(score_npmi))

    if matric == 'top-nmi':
        labels = np.array(df['train']['label'])
        doc_topic_dist = model_output['topic-document-matrix'].T
        max_topic_indices = np.argmax(doc_topic_dist, axis=1)
        score_tn = normalized_mutual_info_score(labels, max_topic_indices, average_method='arithmetic')
        print('Top-NMI:', "{:.4f}".format(score_tn))

    if matric == 'km-nmi':
        labels = np.array(df['train']['label'])
        doc_topic_dist = model_output['topic-document-matrix'].T
        kmeans = KMeans(n_clusters=tm.numb_embeddings, random_state=SEED).fit(doc_topic_dist)
        km_labels = kmeans.labels_
        score_kn = normalized_mutual_info_score(labels, km_labels, average_method='arithmetic')
        print('Km-NMI:', "{:.4f}".format(score_kn))

    if matric == 'perplexity':
        score_perplexity = tm.calculate_perplexity()
        print("perplexity:", "{:.4f}".format(score_perplexity))


parser = argparse.ArgumentParser(description='vMNVTM')

parser.add_argument('--Lambda', type=float, default=1e-8, help='Lambda')
parser.add_argument('--temp', type=int, default=10, help='Temperature')
parser.add_argument('--kappa1', type=int, default=50, help='kappa in encoder')
parser.add_argument('--kappa2', type=int, default=50, help='kappa in decoder')
parser.add_argument('--n_topics', type=int, default=20, help='Number of topics')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--metrics', nargs='+', default=['c_v'], help='Evaluation metrics')

if __name__ == '__main__':
    args = parser.parse_args()
    SEED = 42
    set_seed(SEED)
    df = load_dataset("csv", data_files={'train': './data/20news_train.csv',
                                         'test': './data/20news_test.csv'})

    tm, model_output = train(gamma=args.Lambda, temp=args.temp, kappa1=args.kappa1, kappa2=args.kappa2, n_topics=args.n_topics, n_epochs=args.epochs, lambda_sh=0.1)
    for metric in args.metrics:
        evaluate(tm, model_output, metric)

