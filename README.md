Deep Learning for Virtual Screening
===================================

This template script can be used for benchmarking. 

Dependency
----------

- Keras 1.1.0 or later
  https://github.com/fchollet/keras  

- intel Theano 0.9.0dev or later

- Anaconda 4.2.0

- Python 3.5.2

Files
-----

- main.py
Template script for parameter search.

- dnn.py
Main functions of DNN.

- cpi.npz  
Sample data for predicting compound protein interactions.  
Download from https://my.syncplicity.com/share/vvks9oqxas1xneg/cpi

Usage
-----

    $ python benchmark.py cpi.npz
