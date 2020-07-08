Deep Learning for Multi-modal Virtual Screening
===============================================

Dependency
----------

- Keras-2.0 or later

Usage
-----

```
$ python main.py -h

usage: main.py [-h] [--datafile DATAFILE] [--modelfile MODELFILE]
               [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
               [--dropout DROPOUT] [--random_seed RANDOM_SEED]

optional arguments:
  -h, --help            show this help message and exit
  --datafile DATAFILE
  --modelfile MODELFILE
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR
  --dropout DROPOUT
  --random_seed RANDOM_SEED
```

Results
-------

```
Epoch 1/300
11/11 [==============================] - 2s 170ms/step - loss: 0.7024 - val_loss: 0.6934
Epoch 2/300
11/11 [==============================] - 1s 78ms/step - loss: 0.6929 - val_loss: 0.6922
Epoch 3/300
11/11 [==============================] - 1s 76ms/step - loss: 0.6912 - val_loss: 0.6906
Epoch 4/300
11/11 [==============================] - 1s 76ms/step - loss: 0.6894 - val_loss: 0.6896
Epoch 5/300
11/11 [==============================] - 1s 76ms/step - loss: 0.6854 - val_loss: 0.6866
Epoch 6/300
11/11 [==============================] - 1s 82ms/step - loss: 0.6827 - val_loss: 0.6834
Epoch 7/300
11/11 [==============================] - 1s 74ms/step - loss: 0.6803 - val_loss: 0.6807
Epoch 8/300
11/11 [==============================] - 1s 74ms/step - loss: 0.6763 - val_loss: 0.6763
Epoch 9/300
11/11 [==============================] - 1s 74ms/step - loss: 0.6701 - val_loss: 0.6734
Epoch 10/300
11/11 [==============================] - 1s 78ms/step - loss: 0.6643 - val_loss: 0.6658
...

```
