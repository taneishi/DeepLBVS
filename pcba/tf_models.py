"""
Script that trains Tensorflow multitask models on PCBA dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import deepchem as dc
from pcba_datasets import load_pcba
import timeit

np.random.seed(123)

pcba_tasks, pcba_datasets, transformers = load_pcba()
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
    dropouts=[.25],
    learning_rate=0.001, weight_init_stddevs=[.1],
    batch_size=64, verbosity="high")

start = timeit.default_timer()

# Fit trained model
model.fit(train_dataset)

train_time = timeit.default_timer() - start

model.save()

#Use AUC classification metric
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean, mode="classification")

start = timeit.default_timer()

train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

eval_time = timeit.default_timer() - start

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

print("Test scores")
print(test_scores)

print('Train time: %.1fm' % (train_time/60.))
print('Eval time: %.1fm' % (eval_time/60.))
