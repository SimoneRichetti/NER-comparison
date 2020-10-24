# -*- coding: utf-8 -*-
"""
Utility functions for ML models.

@author: Simone Richetti <srichetti@expertsystem.com>
"""
import time
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def compute_prediction_latency(dataset, model, n_instances=-1):
    """Compute prediction latency of a model.
    
    The model must have a predict method.
    """
    if n_instances == -1:
        n_instances = len(dataset)
    start_time = time.process_time()
    model.predict(dataset)
    total_latency = time.process_time() - start_time
    return total_latency / n_instances


def get_benchmark_crf():
    """One CRF configuration to rule them all. - Sauron"""
    crf = sklearn_crfsuite.CRF(
        algorithm = 'lbfgs',
        c1 = 0.1,
        c2 = 0.5,
        max_iterations = 800,
        all_possible_transitions = True,
        verbose = True
    )
    return crf


def get_crf_gridsearch(output_labels, n_jobs=4):
    '''Create a GridSearchCV onject that search the best CRF configuration'''
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        all_possible_transitions=True,
        verbose=True
    )
    params_space = {
        'c1': [0, .1, .4],
        'c2': [.1, .5, 1],
        'max_iterations': [200, 400, 800],
    }

    labels = list(output_labels.copy())
    labels.remove('O')
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    gs = GridSearchCV(crf, params_space, cv=3, n_jobs=n_jobs,
                      verbose=1, scoring=f1_scorer)
    return gs


def from_encode_to_literal_labels(y_true, y_pred, idx2tag):
    '''Transform sequences of encoded labels in sequences of string labels'''
    let_y_true = list()
    let_y_pred = list()
    for sent_idx in range(len(y_true)):
        let_sent_true = []
        let_sent_pred = []
        for token_idx in range(len(y_true[sent_idx])):
            let_sent_true.append(idx2tag[y_true[sent_idx][token_idx]])
            let_sent_pred.append(idx2tag[y_pred[sent_idx][token_idx]])
        let_y_true.append(let_sent_true)
        let_y_pred.append(let_sent_pred)
    
    return let_y_true, let_y_pred
