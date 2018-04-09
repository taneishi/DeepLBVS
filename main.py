import numpy as np
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
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    data = np.nan_to_num(data)
    print(data)
    np.random.shuffle(data)
    print(str(data.shape))

    taskname = os.path.basename(datafile)
    optimizer = 'Adam'
    lr = 0.0001
    epochs = 100
    dropout = 0.1
    batch_size = 1500

    for activation in ['relu']:
        for unit1 in [3000]:
            for unit2 in [50]:
                dnn.validation(taskname, data, layers=[unit1, unit2], 
                        batch_size=batch_size, epochs=epochs, class_weight=None,
                        optimizer=optimizer, lr=lr, activation=activation,
                        dropout=dropout, patience=100, count=1)
