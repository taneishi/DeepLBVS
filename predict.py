import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os

def main():
    np.random.seed(123)
    
    data = np.load('data/pcba.npz')
    X = data['X']
    y = data['y']
    print(X.shape)

    cls = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(cls, X, y, cv=5, scoring='roc_auc')
    print(scores.mean())

if __name__ == '__main__':
    main()
