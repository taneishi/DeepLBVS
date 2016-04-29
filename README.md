Deep Learning for Virtual Screening
===================================

This template script can be used for benchmarking. 

Dependency
----------

- Keras 1.0.1
  https://github.com/fchollet/keras
  commit b1e47f7741cf526cb3381c4944f20582f368ba27

- intel Theano (0.80rc fork or later for work with Keras 1.0.1)

- intel Numpy (for intel Theano)

- pandas 0.15 (can be older version)
- scikit-learn 0.17 (can be older version)

Files
-----

- mlp.py
  Template script for parameter search.

- cpi.npz
  Sample data for predicting compound protein interactions.
  Download from https://my.syncplicity.com/share/vvks9oqxas1xneg/cpi

Usage
-----

$ python
>>> import numpy as np
>>> np.load('cpi.npz')
array([[ 0.20235342,  0.13306834,  0.23337506, ...,  0.2016128 ,
         0.2054604 ,  1.        ],
       [ 0.15328281,  0.18236028,  0.15194339, ...,  0.08339998,
         0.11678988,  1.        ],
       [ 0.30227754,  0.05331679,  0.3800444 , ...,  0.04803333,
         0.04840195,  1.        ],
       ...,
       [ 0.3487972 ,  0.21659626,  0.3485468 , ...,  0.15965709,
         0.18344697,  0.        ],
       [ 0.20419115,  0.22017394,  0.20193382, ...,  0.07087177,
         0.06966367,  0.        ],
       [ 0.22705786,  0.16934164,  0.26198766, ...,  0.24239591,
         0.18216231,  0.        ]], dtype=float32)
>>> CTRL-D

$ python mlp.py cpi.npz

... loading data
(248195, 1975)
Train on 198556 samples, validate on 49639 samples
Epoch 1/200
198556/198556 [==============================] - 10s - loss: 0.9153 - acc:
0.4989 - val_loss: 0.6943 - val_acc: 0.4991
Epoch 2/200
198556/198556 [==============================] - 10s - loss: 0.6956 - acc:
0.5057 - val_loss: 0.6913 - val_acc: 0.5193
Epoch 3/200
198556/198556 [==============================] - 10s - loss: 0.6938 - acc:
0.5176 - val_loss: 0.6873 - val_acc: 0.5522
Epoch 4/200
198556/198556 [==============================] - 10s - loss: 0.6858 - acc:
0.5449 - val_loss: 0.6736 - val_acc: 0.5755
Epoch 5/200
198556/198556 [==============================] - 10s - loss: 0.6647 - acc:
0.5925 - val_loss: 0.6530 - val_acc: 0.6176
Epoch 6/200
198556/198556 [==============================] - 10s - loss: 0.6297 - acc:
0.6433 - val_loss: 0.6139 - val_acc: 0.6679
Epoch 7/200
198556/198556 [==============================] - 10s - loss: 0.5945 - acc:
0.6775 - val_loss: 0.5755 - val_acc: 0.7082
Epoch 8/200
198556/198556 [==============================] - 10s - loss: 0.5670 - acc:
0.7028 - val_loss: 0.5496 - val_acc: 0.7200
Epoch 9/200
198556/198556 [==============================] - 10s - loss: 0.5339 - acc:
0.7338 - val_loss: 0.5184 - val_acc: 0.7486
Epoch 10/200
198556/198556 [==============================] - 10s - loss: 0.5111 - acc:
0.7522 - val_loss: 0.4948 - val_acc: 0.7696

...

Epoch 190/200
198556/198556 [==============================] - 10s - loss: 0.0619 - acc:
0.9778 - val_loss: 0.2891 - val_acc: 0.9099
Epoch 191/200
198556/198556 [==============================] - 10s - loss: 0.0604 - acc:
0.9785 - val_loss: 0.2965 - val_acc: 0.9124
Epoch 192/200
198556/198556 [==============================] - 10s - loss: 0.0581 - acc:
0.9797 - val_loss: 0.2971 - val_acc: 0.9094
Epoch 193/200
198556/198556 [==============================] - 10s - loss: 0.0593 - acc:
0.9794 - val_loss: 0.3098 - val_acc: 0.9051
Epoch 194/200
198556/198556 [==============================] - 10s - loss: 0.0578 - acc:
0.9795 - val_loss: 0.2850 - val_acc: 0.9153
Epoch 195/200
198556/198556 [==============================] - 10s - loss: 0.0630 - acc:
0.9768 - val_loss: 0.2930 - val_acc: 0.9175
Epoch 196/200
198556/198556 [==============================] - 10s - loss: 0.0546 - acc:
0.9812 - val_loss: 0.2965 - val_acc: 0.9177
Epoch 197/200
198556/198556 [==============================] - 10s - loss: 0.0547 - acc:
0.9810 - val_loss: 0.3069 - val_acc: 0.9076
Epoch 198/200
198556/198556 [==============================] - 10s - loss: 0.0571 - acc:
0.9801 - val_loss: 0.2966 - val_acc: 0.9176
Epoch 199/200
198556/198556 [==============================] - 10s - loss: 0.0581 - acc:
0.9795 - val_loss: 0.2919 - val_acc: 0.9185
Epoch 200/200
198556/198556 [==============================] - 10s - loss: 0.0541 - acc:
0.9811 - val_loss: 0.2994 - val_acc: 0.9102
Log file saved as result/cpi.npz_3000_1000_adam_200.log
ran for 2179.4s

