# -*- coding: utf-8 -*-
"""
A module for reading data from famous datasets.

@author: Simone Richetti <srichetti@expertsystem.com>
"""

import os
import re
import urllib
import pandas as pd
from math import nan


CONLL_URL_ROOT = "https://raw.githubusercontent.com/davidsbatista/NER-datasets/master/CONLL2003/"


def open_read_from_url(url):
    """
    Take in input an url to a .txt file and return the list of its rows.
    
    Parameters
    ----------
    url : str
        url of the source file

    Returns
    -------
    lines
        a list of strings that are the file rows
    """
    
    print(f"Read file from {url}")
    file = urllib.request.urlopen(url)
    lines = []
    for line in file:
        lines.append(line.decode("utf-8"))

    return lines


def read_raw_conll(url_root, dir_path, filename):
    """Read a file from disk or from url.
    
    Parameters
    ----------
    url_root : str
        url of the source file
        
    dir_path : str
        path of the directory of the source file
    
    filename : str
        name of the source file
        
    Returns
    -------
    lines
        a list of strings that are the file rows
    """
    
    lines = []
    path = os.path.join(dir_path, filename)
    full_url = url_root + filename
    if os.path.isfile(path):
        print(f'Reading file {path}')
        with open(path, 'r') as f:
            lines = f.readlines()
    else:
        lines = open_read_from_url(full_url)
    return lines


def is_real_sentence(only_token, sentence):
    """Chek if a sentence is a real sentence or a document separator.
    
    Parameters
    ----------
    only_token : bool
        if it is true, the sentence is a list of tokens, if false is a list
        of token features
        
    sentence : list
        the sentence we want to check, as list of tokens or list of token 
        features
    
    Returns
    -------
    bool
        False if the sentence is a document separator, True otherwise.
    """
    
    first_word = ""
    if only_token:
        first_word = sentence[0]
    else:
        first_word = sentence[0][0]

    if '---------------------' in first_word or first_word == '-DOCSTART-':
        return False
    else:
        return True
        
        
def load_conll_data(filename, url_root=CONLL_URL_ROOT, dir_path='', 
                    only_tokens=False):
    """
    Read CoNLL03 dataset.
    
    Take an url to the raw .txt files that you can find the repo linked above,
    load data and save it into a list of tuples data structure.
    Those files structure data with a word in each line with word, POS, 
    syntactic tag and entity tag separated by a whitespace. Sentences are 
    separated by an empty line.
    
    Parameters
    ----------
    filename : str
        name of the file which contains a CoNLL03 dataset
        
    url_root : str, optional
        url root where we can download CoNLL03 dataset (default is 
        CONLL_URL_ROOT)
    
    dir_path : str, optional
        path of the CoNLL03 file (default is empty string)
    
    only_tokens : bool, optional
        whether we want to return only tokens or also token features (default 
        is False)
    
    Returns
    -------
    X : list
        list of sentences, each sentence is a list of tokens. A token can be a 
        string or a list of features
    
    Y : list
        list of sentence labels, which are a list of labels of each token
    
    output_labels : set
        set with all the different tags in the dataset
    """
    
    lines = read_raw_conll(url_root, dir_path, filename)[2:]
    # TODO: find a better data structure for saving data
    # TODO: do this in a more efficient way
    X = []
    Y = []
    output_labels=set()
    
    sentence = []  # Temporary data structure
    labels = []    # Temporary data structure
    for line in lines:
        if line == "\n":   # Empty line
            if(len(sentence) != len(labels)):
                print(f"Error: we have {len(sentence)} words but {len(labels)} labels")
            if sentence and is_real_sentence(only_tokens, sentence):
                X.append(sentence)
                Y.append(labels)
            sentence = []
            labels = []
        else:
            features = line.split() # features are separated with whitespaces
            tag = features.pop()    # last element of the feature vector is 
                                    # entity tag
            labels.append(tag)
            output_labels.add(tag)
            if only_tokens:
                sentence.append(features.pop(0))
            else:
                sentence.append(tuple(features))
    
    print(f"Read {len(X)} sentences")
    if(len(X) != len(Y)):
        print("ERROR in reading data.")
    return X, Y, output_labels


# =========================================================================== #


def _df_to_xy(df):
    """Transform ACNER dataframe in X, y sets.
    
    Given the ACNER dataframe, we want to obtain a X list of lists of 
    dictionaries, which contain the features of the tokens, and a Y list of 
    lists of the tag of the tokens.
    
    Parameters
    ----------
    df : pandas.Dataframe
        pandas dataframe which contains ACNER dataset
            
    Returns
    -------
    X : list
        list of lists of dictionaries, which contain token features
        
    y : list
        list of lists of strings, which are token tags
    """
    
    y = df[['sentence_idx', 'tag']].copy()
    y = y.groupby('sentence_idx').apply(lambda d: d['tag'].values.tolist()).values
    
    df.drop(columns='tag', inplace=True)
    X = df.groupby('sentence_idx').apply(lambda d: d.to_dict('records')).values
    if len(X) != len(y):
        print('ERROR: length mismatch')
    else:
        print(f'Dataset dimension: {len(y)} sentences')
    return X, y


def load_anerd_data(path, filter_level=''):
    """Load anerd data from path.
    
    Parameters
    ----------
    path : str
        path of ACNER dataset csv file
            
    filter_level : str, optional
        parameter which indicate what data extract from anerd:
            default         extract features of each token and the 
                                neighbours
            sentence_only   extract only the list of tokens
            all_data        extract features of each token, the neighbours 
                                and neighbours of neighbours
                                
    Returns
    -------
    X : list
        list of lists of dictionaries, which contain token features
        
    y: list
        list of lists of token labels
            
    tags : set 
        set with all the different tags
    """
    
    dframe = pd.read_csv(path, encoding = "ISO-8859-1", error_bad_lines=False)
    
    # Create label set
    tags = set()
    for tag in set(dframe["tag"].values):
        if tag is nan or isinstance(tag, float):
            tags.add('unk')
        else:
            tags.add(tag)

    if filter_level == 'sentence_only':
        # Return only the list of tokens for each sentence
        print('Filter level:', filter_level)
        dframe = dframe[['word', 'sentence_idx', 'tag']]
        X, y = _df_to_xy(dframe)
        newX = []
        for sent in X:
            sentence = []
            for d in sent:
                sentence.append(d['word'])
            newX.append(sentence)
        X = newX
    
    elif filter_level == 'all_data':
        # Return all the features for each token
        print('Filter level:', filter_level)
        dframe.drop(columns=['Unnamed: 0', 'prev-iob', 'prev-prev-iob'], 
                    inplace=True)
        print('Features:', dframe.columns)
        X, y = _df_to_xy(dframe)
    
    else:
        # Return features of the token itself and the features of the 
        # adjacent tokens
        print('Filter level: default')
        dframe.drop(columns= ['Unnamed: 0', 'prev-iob', 'next-next-lemma', 
                              'next-next-pos', 'next-next-shape', 
                              'next-next-word', 'prev-prev-iob', 
                              'prev-prev-lemma','prev-prev-pos', 
                              'prev-prev-shape', 'prev-prev-word'], inplace=True)
        print('Features:', dframe.columns)
        X, y = _df_to_xy(dframe)
    
    print('Data read successfully!')
    return X, y, tags


# =========================================================================== #


def load_wikiner(path, token_only=False):
    """Load WikiNER dataset.
    
    Parameters
    ----------
    path : str
        path to txt file if WikiNER dataset
            
    token_only: bool, optional
        if True return only the list of tokens, else return also pos 
        tag for each token (default is False)
        
    Returns
    -------
    sentences : list
        list of sentences, each sentences is a list of token
            
    tags : list
        list of list of token tags
            
    output_labels : set
        set of all the labels in the dataset
    """
    
    raw_sents = []
    with open(path, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            if line != '\n':
                raw_sents.append(line)
    
    # Split tokens
    for sent_idx in range(len(raw_sents)):
        raw_sents[sent_idx] = raw_sents[sent_idx].split()
    
    # Extract features and separate them from tags
    sentences = []
    tags = []
    output_labels = set()
    for raw_sent in raw_sents:
        sent = []
        tag = []
        for word in raw_sent:
            features = word.split('|')
            ent = features.pop()
            tag.append(ent)
            output_labels.add(ent)
            if token_only:
                sent.append(features.pop(0))
            else:
                sent.append(tuple(features))
        sentences.append(sent)
        tags.append(tag)
    print(f'Read {len(sentences)} sentences.')
    return sentences, tags, output_labels


def _get_digits(text):
    """Preprocess numbers in tokens accordingly to itWac word embedding.
    
    Parameters
    ----------
    text : str
        string of text
    
    Returns
    -------
    str
        preprocessed text
    """
    
    try:
        val = int(text)
    except:
        text = re.sub('\d', '@Dg', text)
        return text
    
    if val >= 0 and val < 2100:
        return str(val)
    else:
        return "DIGLEN_" + str(len(str(val)))


def _normalize_text(word):
    """Preprocess word in order to match with the itWac embedding vocabulary
    
    Parameters
    ----------
    word : string
        string representation of a word
        
    Returns
    -------
    str
        preprocessed word
    """
    
    if "http" in word or ("." in word and "/" in word):
        word = str("___URL___")
        return word
    if len(word) > 26:
        return "__LONG-LONG__"
    new_word = _get_digits(word)
    if new_word != word:
        word = new_word
    if word[0].isupper():
        word = word.capitalize()
    else:
        word = word.lower()
    return word


def itwac_preprocess_data(sentences):
    """Preprocess text in order to match with the itWac embedding vocabulary
    
    Parameters
    ----------
    sentences : list
        list of sentences
    
    Returns
    -------
     new_sentences : list
        list of preprocessed sentences
    """
    
    new_sentences = []
    for sentence in sentences:
        new_sent = list()
        for word in sentence:
            new_sent.append(_normalize_text(word))
        new_sentences.append(new_sent)
    return new_sentences
