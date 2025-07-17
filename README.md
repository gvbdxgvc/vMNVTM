# vNMVTM

This repository includes the source code of "Neural Topic Modeling on Hyperspheres: Spherical Representation Learning with von Mises–Fisher Mixtures"

## Dependency
* python 3.9
* torch 2.1.0
* CUDA 12.5
* scikit-learn 1.2.2
* nltk 3.8.1
* gensim 4.3.0
* spacy 3.5.1
* scipy 1.11.4


## An example to run

```train
python main.py --n_topics 20 --epochs 200 --metrics c_v
```

## Other aruguments

* --Lambda: the weighted hyperparameter of vEC loss
* --temp: the temperature coefficient
* --kappa1: the kappa in the encoder
* --kappa2: the kappa in the decoder
* --n_topics: the number of topics
* --epochs: the number of epochs
* --metrics: evaluation metrics for topic model (c_v, c_npmi, diversity, top-nmi, km-nmi, perplexity, keywords)


## Custom modification
The dataset can be replaced with any custom one in main.py with the format below:

| text                             | label | label name    |
|----------------------------------|-------|---------------|
| (Unnecessary to be preprocessed) |       | (Unnecessary) |

The pretrain word embedding can be easily changed from HuggingFace:
```arg
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
```

