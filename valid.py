from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import preprocessing
import glob
import os

def main():
    for filename in sorted(glob.glob('dragon/AID*')):
        data, target = load_svmlight_file(filename)
        data = data.todense()
        data = preprocessing.minmax_scale(data)

        rfc = RandomForestClassifier(n_estimators=100)

        scores = cross_validation.cross_val_score(
                rfc, data, target, cv=4, scoring='roc_auc')
        print os.path.basename(filename), scores.mean()

if __name__ == '__main__':
    main()
