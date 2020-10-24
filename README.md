# Comparison of Entity Extraction techniques for annotation

This repo contains the implementation of several Machine Learning algorithms for Named Entity Recognition.
We build, train and evaluate them on many different dataset, considering several aspects: quality of prediction, memory consumption, and latency of inference.

## Dependencies
See `environment.yml`. In general, I used `tensorflow.keras` and `scikit-learn` for my ML experiments :crystal_ball:.

## Setup
```bash
conda env create -f environment.yml
conda activate ner-suite
```

You can now play with the notebooks!

## Project Structure
* `data/`: directory in which are saved all the dataset used in the notebooks. The dataset are:
    * [CoNLL03](https://paperswithcode.com/sota/named-entity-recognition-ner-on-conll-2003);
    * [Annotated Corpus for NER](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus);
    * [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner) (english and italian);
* `embeddings/`: directory that contains different word embeddings:
    * `glove.6B.100d.txt` for english;
    * `w2v.itWac.128d.txt` for italian;
* `utils`: a package that I made in order to increase code modularity, reusability and readability;
* `<algo>-<dataset>.ipynb`: these are the notebooks with the experiments that we made;
* `environment.yml`: conda environment file in order to replicate the environment on your machine and reproduce the experiments;
* `results.xlsx`: results of the experiments;

## Models and references:
* **Conditional Random Fields:** a traditional Machine Learning algorithm which can deal with sequences. Refer to the [original paper](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers) and the implementation of the [sklearn wrapper](https://sklearn-crfsuite.readthedocs.io/en/latest/);
* **LSTM:** the most used *recurrent neural network* for modeling sequences. We also use it in combination with pre-trained embeddings like [GloVe](https://nlp.stanford.edu/projects/glove/) and [itWac](http://www.italianlp.it/resources/italian-word-embeddings/);
* **End-to-end model:** in [this paper](https://www.aclweb.org/anthology/P16-1101.pdf) it is proposed a model which combines a CNN to extract morphological features from the characters of the word, the GloVe embeddings to represent word-level features, a Bidirectional LSTM to model the context and finally a CRF layer to decode the best sequence of labels. We implemented it, thanks to the work already done in [this repo](https://github.com/napsternxg/DeepSequenceClassification/blob/master/model.py).  


## TODO
* Improve documentation;
