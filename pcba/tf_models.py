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

np.random.seed(123)

pcba_tasks, pcba_datasets, transformers = load_pcba(
        featurizer='ECFP', split='random')
(train_dataset, valid_dataset, test_dataset) = pcba_datasets

print("PCBA_tasks")
print(len(pcba_tasks))
print("Number of compounds in train set")
print(len(train_dataset))
print("Number of compounds in validation set")
print(len(valid_dataset))
print("Number of compounds in test set")
print(len(test_dataset))

model = dc.models.TensorflowMultiTaskClassifier(
    len(pcba_tasks), train_dataset.get_data_shape()[0],
    layer_sizes=[1500], bias_init_consts=[1.], dropouts=[0.5],
    penalty=0.1, penalty_type='l2',
    learning_rate=0.001, weight_init_stddevs=[0.02],
    batch_size=50, verbosity="high")

start = timeit.default_timer()

# Fit trained model
model.fit(train_dataset, nb_epoch=10)

train_time = timeit.default_timer() - start

model.save()

#Use AUC classification metric
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")

start = timeit.default_timer()

train_score, train_scores = model.evaluate(train_dataset, [metric], transformers, per_task_metrics=True)
valid_score, valid_scores = model.evaluate(valid_dataset, [metric], transformers, per_task_metrics=True)
test_score, test_scores = model.evaluate(test_dataset, [metric], transformers, per_task_metrics=True)

eval_time = timeit.default_timer() - start

print("Train scores")
print(train_score)

print("Validation scores")
print(valid_score)

print("Test scores")
print(test_score)

if not os.path.exists('log/pcba'): os.makedirs('log/pcba')
out = open('log/pcba/tf_models.log', 'w')
out.write('Train scores: %s\n' % train_score)
out.write('Validation scores: %s\n' % valid_score)
out.write('Test scores: %s\n' % test_score)
out.write('Train time: %.1fm\n' % (train_time/60.))
out.write('Eval time: %.1fm\n' % (eval_time/60.))
out.close()

scores = [
        train_scores['mean-roc_auc_score'],
        valid_scores['mean-roc_auc_score'],
        test_scores['mean-roc_auc_score'],
        ]
scores = pd.DataFrame(scores).T
scores.columns = ['train','valid','test']
scores.to_pickle('log/pcba/tf_models.pkl')
