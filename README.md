# Ligand-Based Virtual Screening using Deep Learning

## Usage

```
bash run.sh
```

will generate ECFP fingerprints for each PCBA dataset and
perform 5-fold CV of RandomForest model for assay results.

```
              count  negative  positive  percentage
PCBA-884      10471      7048      3423       32.69
PCBA-899       8256      6419      1837       22.25
PCBA-686978  303167    240040     63127       20.82
PCBA-891       7888      6309      1579       20.02
PCBA-686979  309966    260836     49130       15.85
PCBA-883       8158      6921      1237       15.16
PCBA-504332  297588    266903     30685       10.31
PCBA-1030    161832    145842     15990        9.88
PCBA-720532   12987     11981      1006        7.75
PCBA-588342  326954    301893     25061        7.66
 
AID PCBA-884 (  1/ 10)
          count  negative  positive  percentage
PCBA-884  10471      7048      3423       32.69
RandomForestClassifier(n_estimators=200) 5-fold CV mean AUC 0.834 28.250sec

AID PCBA-899 (  2/ 10)
          count  negative  positive  percentage
PCBA-899   8256      6419      1837       22.25
RandomForestClassifier(n_estimators=200) 5-fold CV mean AUC 0.733 21.159sec

...
```

## Results of ECFP4 1024 bits

```
Resluts of ecfp_4_1024
                  rf     xgb     mlp
PCBA-884     0.8340  0.8579  0.9060
PCBA-899     0.7333  0.7552  0.8347
PCBA-686978  0.6843  0.7324  0.8879
PCBA-891     0.7470  0.7853  0.8599
PCBA-686979  0.6499  0.6987  0.8658
PCBA-883     0.7009  0.7353  0.7933
PCBA-504332  0.6358  0.6548  0.7726
PCBA-1030    0.6059  0.6352  0.6998
PCBA-720532  0.6526  0.6922  0.5623
PCBA-588342  0.6936  0.7341  0.8667
MeanAUC      0.6937  0.7281  0.8049
```
