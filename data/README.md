# Dataset
In this directory we store the datasets on which we test the different ML models.


## CoNLL03
[CoNLL03](https://www.aclweb.org/anthology/W03-0419/) is one of the most used dataset for NER models evaluation.
The notebooks look for a directory `conll03` with three files within: `train.txt`, `test.txt` and `valid.txt`.

You can find the dataset in many GitHub repositories, like [this ](https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003) (if the notebook does not find the directory, it will automatically try to download it from that repo).


## Annotated Corpus for Named Entity Recognition
This dataset is taken from Kaggle: you can download it from the relative [page](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus).

The notebooks look for a directory called `annotated-ner-dataset` with a `ner.csv` file within it.


## WikiNER
[WikiNER](https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500) is a dataset composed of annotated Wikipedia pages.

The notebooks look for two files, `wikiner-en-wp3-raw.txt` and `wikiner-it-wp3-raw.txt`, in this `data` directory. 
You can make an official request or you can take them from [this repo ](https://github.com/dice-group/FOX/tree/master/input/Wikiner) ([english](https://github.com/dice-group/FOX/blob/master/input/Wikiner/aij-wikiner-en-wp3.bz2), [italian](https://github.com/dice-group/FOX/blob/master/input/Wikiner/aij-wikiner-it-wp3.bz2)).