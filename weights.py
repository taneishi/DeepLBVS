import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(filename):
    params = np.load(filename)['arr_0'].item()
    weights = filter(lambda x: x.startswith('W'), params.keys())
    print(weights)
    plt.figure(figsize=(16,9))
    for i,key in enumerate(sorted(weights), 1):
        if key.startswith('W'):
            weight = params[key]
            layer = int(key[1:])
            ax = plt.subplot(1, len(weights), i)
            im = ax.imshow(weight, aspect='auto')
            plt.colorbar(im)
            if layer == len(weights)-1:
                plt.xlabel('Output')
            else:
                plt.xlabel('Layer%d' % (layer+1))
            if layer == 0:
                plt.ylabel('Input')
            else:
                plt.ylabel('Layer%d' % layer)
            plt.title(key)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('weights [filename]')
    main(sys.argv[1])
