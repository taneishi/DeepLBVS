"""
Script that trains Tensorflow multitask models on PCBA dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd
import deepchem as dc
from datasets import load_pcba
import timeit

from deepchem.data.datasets import DiskDataset
from sklearn.model_selection import KFold

np.random.seed(123)

pcba_tasks, pcba_datasets, transformers = load_pcba(
        featurizer='ECFP', split='random')
(train_dataset, valid_dataset, test_dataset) = pcba_datasets

print("PCBA_tasks")
print(len(pcba_tasks))
print("Number of compounds in train set")
print(len(train_dataset))
print("Number of compounds in test set")
print(len(test_dataset))

X = train_dataset.X
y = train_dataset.y
w = train_dataset.w

kf = KFold(n_splits=5, shuffle=True, random_state=123)

all_train_scores = []
all_test_scores = []

start = timeit.default_timer()

for train_index, test_index in kf.split(X):

    train_dataset = DiskDataset.from_numpy(X[train_index,:], y[train_index, :], w[train_index, :], verbose=False)
    test_dataset = DiskDataset.from_numpy(X[test_index,:], y[test_index, :], w[test_index, :], verbose=False)

    model = dc.models.TensorflowMultiTaskClassifier(
        len(pcba_tasks), train_dataset.get_data_shape()[0],
        layer_sizes=[1500], bias_init_consts=[1.], dropouts=[0.5],
        penalty=0.1, penalty_type='l2',
        learning_rate=0.001, weight_init_stddevs=[0.02],
        batch_size=50, verbosity="high")

    # Fit trained model
    model.fit(train_dataset, nb_epoch=10)

    #Use AUC classification metric
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")

    train_score, train_scores = model.evaluate(train_dataset, [metric], transformers, per_task_metrics=True)

    print("Train scores")
    print(train_score)

    test_score, test_scores = model.evaluate(test_dataset, [metric], transformers, per_task_metrics=True)

    print("Test scores")
    print(test_score)

    all_train_scores.append(train_scores['mean-roc_auc_score'])
    all_test_scores.append(test_scores['mean-roc_auc_score'])

train_time = timeit.default_timer() - start

all_train_scores = np.concatenate(all_train_scores)
all_test_scores = np.concatenate(all_test_scores)

if not os.path.exists('log/pcba'): os.makedirs('log/pcba')
out = open('log/pcba/tf_models.log', 'w')
out.write('Train scores: %s\n' % np.mean(all_train_scores))
out.write('Test scores: %s\n' % np.mean(all_test_scores))
out.write('Train time: %.1fm\n' % (train_time/60.))
out.close()

scores = [
        all_train_scores,
        all_test_scores,
        ]
scores = pd.DataFrame(scores).T
scores.columns = [
        'train',
        'test']
scores.to_pickle('log/pcba/tf_models.pkl')
