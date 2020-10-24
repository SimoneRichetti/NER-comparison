# -*- coding: utf-8 -*-
"""
A module with utility functions for machine learning models.

@author: Simone Richetti <srichetti@expertsystem.com>
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional # , TimeDistributed
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD


def get_model_memory_usage(batch_size, model):
    """Return memory usage of a model in MB given the batch size"""
    import numpy as np
    try:
        from tensorflow.keras import backend as K
    except:
        from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    mbytes = np.round(total_memory / (1024.0 ** 2), 3) + internal_model_mem_count
    return mbytes


def print_model_memory_usage(batch_size, model):
    """Print memory usage of a model in MB given the batch size"""
    mbytes = get_model_memory_usage(batch_size, model)
    print(f'Model size: {mbytes} MB')


def load_glove_embedding_matrix(path, word_index, embed_dim):
    """Load Glove embeddings.
    
    More info here: 
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    """
    embeddings_index = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


def load_w2v_nlpl_embedding_matrix(path, word_index, embed_dim):
    """Load NLPL Italian embedding."""
    embeddings_index = {}
    with open(path, encoding='iso-8859-1') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                continue
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix


def create_BiLSTM(vocabulary_size, seq_len, n_classes, hidden_cells=128, 
                  embed_dim=32, drop=0.5, use_glove=False, glove_matrix=None):
    """Create a BiLSTM model using keras, given its parameters"""
    model = Sequential()
    if use_glove:
        model.add(Embedding(vocabulary_size, embed_dim, 
                            weights=[glove_matrix], input_length=seq_len,
                            mask_zero=True, trainable=True))
    else:
        model.add(Embedding(vocabulary_size, embed_dim, input_length=seq_len, 
                            mask_zero=True))
    model.add(Dropout(drop))
    
    model.add(Bidirectional(LSTM(hidden_cells, return_sequences=True, 
                                 dropout=drop, recurrent_dropout=drop)))
    model.add(Bidirectional(LSTM(hidden_cells, return_sequences=True, 
                                 dropout=drop, recurrent_dropout=drop)))
    
    model.add(Dense(hidden_cells))
    model.add(Dropout(drop))
    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy',
                           Precision(),
                           Recall()])
    model.summary()
#     plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


def create_paper_BiLSTM(vocabulary_size, seq_len, n_classes, hidden_cells=200, 
                  embed_dim=100, drop=0.4, use_glove=False, glove_matrix=None):
    """Create a BiLSTM model using keras, given its parameters"""
    model = Sequential()
    if use_glove:
        model.add(Embedding(vocabulary_size, embed_dim, 
                            weights=[glove_matrix], input_length=seq_len,
                            mask_zero=True, trainable=True))
    else:
        model.add(Embedding(vocabulary_size, embed_dim, input_length=seq_len, 
                            mask_zero=True))
    model.add(Dropout(drop))
    
    model.add(Bidirectional(LSTM(hidden_cells, return_sequences=True, 
                                 dropout=drop)))

    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
#                   optimizer=SGD(learning_rate=0.015, momentum=0.9, clipvalue=5.),  # decay rate missing
                  metrics=['accuracy',
                           Precision(),
                           Recall()])
    model.summary()
#     plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


def remove_flat_padding(X, y_true, y_pred, pad=0):
    """Remove padding predictions and flatten the list of sequences"""
    new_true = []
    new_pred = []
    y_true_flat = np.array(y_true).ravel()
    y_pred_flat = np.array(y_pred).ravel()
    X_flat = np.array(X).ravel()
    for idx in range(len(X_flat)):
        if X_flat[idx] != pad:
            new_true.append(y_true_flat[idx])
            new_pred.append(y_pred_flat[idx])
    return np.array(new_true), np.array(new_pred)


def remove_seq_padding(X, y_true, y_pred, pad=0):
    """Remove padding predictions from list of sequences"""
    new_true = []
    new_pred = []
    for sent_idx in range(len(X)):
        true_sent = []
        pred_sent = []
        for tok_idx in range(len(X[sent_idx])):
            if X[sent_idx][tok_idx] != pad:
                true_sent.append(y_true[sent_idx][tok_idx])
                pred_sent.append(y_pred[sent_idx][tok_idx])
        new_true.append(true_sent)
        new_pred.append(pred_sent)
    return np.array(new_true), np.array(new_pred)