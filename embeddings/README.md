# Embeddings
In this folder we store the embeddings needed in the notebooks.

## GloVe
You can download [GloVe](https://nlp.stanford.edu/projects/glove/) at [this link](http://nlp.stanford.edu/data/glove.6B.zip) (GloVe 6B). 
Then, unzip the downloaded archive and copy in this directory the `glove.6B.100d.txt` file.

## itWac
You need to make request for the italian itWac embedding at [this link](http://www.italianlp.it/download-itwac-word-embeddings/). 
After that, unzip the downloaded archive, copy the `itwac128.sqlite` file in this folder and run the command:
```bash
> python conv_script.py itwac128.sqlite w2v.itWac.128d.txt
```