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
 
               4_1024
PCBA-884     0.834002
PCBA-899     0.733279
PCBA-686978  0.684328
PCBA-891     0.746985
PCBA-686979  0.649927
PCBA-883     0.700886
PCBA-504332  0.635792
PCBA-1030    0.605897
PCBA-720532  0.652586
PCBA-588342  0.693636
MeanAUC      0.693732

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
