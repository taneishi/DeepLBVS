# Ligand-Based Virtual Screening using Deep Learning

## Usage

```
bash run.sh
```

will generate ECFP fingerprints for each PCBA dataset and
perform 5-fold CV of RandomForest model for assay results.

```
                count  negative  positive  percentage
PCBA-884      10471.0    7048.0    3423.0   32.690287
PCBA-899       8256.0    6419.0    1837.0   22.250484
PCBA-686978  303167.0  240040.0   63127.0   20.822517
PCBA-891       7888.0    6309.0    1579.0   20.017748
PCBA-686979  309966.0  260836.0   49130.0   15.850125
PCBA-883       8158.0    6921.0    1237.0   15.163030
PCBA-504332  297588.0  266903.0   30685.0   10.311236
PCBA-1030    161832.0  145842.0   15990.0    9.880617
PCBA-720532   12987.0   11981.0    1006.0    7.746208
PCBA-588342  326954.0  301893.0   25061.0    7.664993


AID PCBA-884
Converted compounds 10471/10471
The shape of fingerprints matrix (10471, 2048)
RandomForest 5-fold CV mean AUC 0.934

AID PCBA-899
Converted compounds  8256/ 8256
The shape of fingerprints matrix (8256, 2048)
RandomForest 5-fold CV mean AUC 0.871

...
```
