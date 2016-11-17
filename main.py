import pandas as pd
import numpy as np
from keras.optimizers import Adam
import sys
import os
import dnn

if __name__ == '__main__':
    if len(sys.argv) > 1:
        datafile = sys.argv[1]
    else:
        sys.exit('Usage: %s [datafile]' % (sys.argv[0]))

    for dirname in ['model','result']:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    np.random.seed(123)

    dnn.setup()
    dnn.show_version()

    print('Data loading ...')
    data = np.load(datafile)['data']
    np.random.shuffle(data)
    print(str(data.shape))

    taskname = os.path.basename(datafile)
    optimizer = Adam
    lr = 0.0001
    nb_epoch = 500
    for activation in ['sigmoid']:
        for batch_size in [1500]:
            for unit1 in [3000]:
                for unit2 in [60]:
                    dnn.validation(taskname, data, layers=[unit1, unit2], 
                            batch_size=batch_size, nb_epoch=nb_epoch, 
                            optimizer=optimizer, lr=lr, activation=activation,
                            dropout=0, patience=100, count=5)
