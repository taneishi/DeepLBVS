import numpy as np
import argparse
from sklearn.preprocessing import minmax_scale
from model import validation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='cpi.npz')
    args = parser.parse_args()

    np.random.seed(123)

    print('Data loading ...')
    data = np.load(args.datafile, allow_pickle=True)['data']
    data = minmax_scale(data)
    print(data)

    np.random.shuffle(data)
    print(data.shape)

    lr = 0.0001
    epochs = 1000
    dropout = 0.1
    batch_size = 1500

    validation(data, layers=[3000, 50], 
            batch_size=batch_size, epochs=epochs,
            lr=lr, activation='relu', dropout=dropout)
