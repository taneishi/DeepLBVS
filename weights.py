import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(filename):
    params = np.load(filename)['weights'].item()

    weights = [x for x in params.keys() if x.startswith('W')]
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
    main(sys.argv[1])
