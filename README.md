Prediction of Compound-Protein Interactions
===========================================

Environments
------------

```
python3 -m venv rdkit
source rdkit/bin/activate
pip install --upgrade pip
pip install rdkit-pypi torch pandas scikit-learn
```

Usage
-----

```
python main.py -h
python main.py --epochs 100
```

```
python pcba.py
python pcba.py --limit 10 --sort
```

will generate ECFP fingerprints for each PCBA dataset and
perform 5-fold CV of RandomForest model for assay results.

```
             AID   count  negative  positive    diff
111     PCBA-884   10471      7048      3423    3625
115     PCBA-899    8256      6419      1837    4582
114     PCBA-891    7888      6309      1579    4730
110     PCBA-883    8158      6921      1237    5684
121     PCBA-915    8051      7610       441    7169
..           ...     ...       ...       ...     ...
70   PCBA-602310  394179    393869       310  393559
73   PCBA-624170  398628    397791       837  396954
107  PCBA-743266  399106    398800       306  398494
75   PCBA-624173  401002    400511       491  400020
72   PCBA-602332  413314    413243        71  413172

[128 rows x 5 columns]

AID PCBA-884
Converted compounds 10471/10471
The shape of fingerprints matrix (10471, 2048)
RandomForest 5-fold CV mean AUC 0.934

AID PCBA-899
Converted compounds  8256/ 8256
The shape of fingerprints matrix (8256, 2048)
RandomForest 5-fold CV mean AUC 0.871

AID PCBA-891
Converted compounds  7888/ 7888
The shape of fingerprints matrix (7888, 2048)
RandomForest 5-fold CV mean AUC 0.905

AID PCBA-883
Converted compounds  8158/ 8158
The shape of fingerprints matrix (8158, 2048)
RandomForest 5-fold CV mean AUC 0.876

AID PCBA-915
Converted compounds  8051/ 8051
The shape of fingerprints matrix (8051, 2048)
RandomForest 5-fold CV mean AUC 0.872

AID PCBA-914
Converted compounds  7831/ 7831
The shape of fingerprints matrix (7831, 2048)
RandomForest 5-fold CV mean AUC 0.924

AID PCBA-720532
Converted compounds 12987/12987
The shape of fingerprints matrix (12987, 2048)
RandomForest 5-fold CV mean AUC 0.901

AID PCBA-885
Converted compounds 12846/12846
The shape of fingerprints matrix (12846, 2048)
RandomForest 5-fold CV mean AUC 0.904

AID PCBA-493208
Converted compounds 41636/41636
The shape of fingerprints matrix (41636, 2048)
RandomForest 5-fold CV mean AUC 0.839

AID PCBA-904
Converted compounds 50960/50960
The shape of fingerprints matrix (50960, 2048)
RandomForest 5-fold CV mean AUC 0.814
             AID  count  negative  positive   diff   MeanAUC
111     PCBA-884  10471      7048      3423   3625  0.933937
115     PCBA-899   8256      6419      1837   4582  0.870803
114     PCBA-891   7888      6309      1579   4730  0.904869
110     PCBA-883   8158      6921      1237   5684  0.876085
121     PCBA-915   8051      7610       441   7169  0.871634
120     PCBA-914   7831      7611       220   7391  0.923916
96   PCBA-720532  12987     11981      1006  10975  0.900595
112     PCBA-885  12846     12681       165  12516  0.904030
45   PCBA-493208  41636     41294       342  40952  0.839447
118     PCBA-904  50960     50432       528  49904  0.813762
```
