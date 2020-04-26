# Cascade Inference with LSTM

1. [Overview](#overview)
2. [Prerequisite](#prerequisite)
3. [Result](#result)

## Overview

The goal of this deep learning project is to predict future reservoir production performance based on production history.
Since limited history production data were given, it was not able to apply simple RNN network. Here, we introduced cascade inference method for prediction.


## Prerequisite

 - The Brugge Oil field dataset is used. However, it is not included in this repository.  
 
 | Timestep | WOPR | WBHP | WWCT | WWPR |
 | :---: | ---: | ---: | ---: | ---: |
 |0   |    0.0000 | 2348.024 |  0.000000  |    0.00000 |
 |1   | 1963.8990 | 2104.888 |  0.017938  |   35.87136 |
 |2   | 1962.6620 | 2091.095 |  0.018556  |   37.10804 |
 | ... | ... | ... | ... | ... | 
 |495 |  663.2374 | 1621.689 |  0.668343  | 1336.53300 |
 |496 |  662.5985 | 1621.971 |  0.668663  | 1337.17200 |
 |497 |  661.1443 | 1622.610 |  0.669390  | 1338.62600 |
 
 - each column stands for
     - `WOPR`: Well Oil Production Rate, which will be target of prediction
     - `WBHP`: Well Bottom Hole Pressure
     - `WWCT`: Well Water Cut
     - `WWPR`: Well Water Production Rate

## Result

The accuracy of `WOPR` prediction was vary from well to well.

<div align="center">
    <img src="https://user-images.githubusercontent.com/13795717/80310120-0a321800-8814-11ea-9a45-c82bfd18b832.png" width="250">
    <img src="https://user-images.githubusercontent.com/13795717/80310194-6e54dc00-8814-11ea-9f41-5e3ce1909270.png" width="250">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/13795717/80310168-4a919600-8814-11ea-92aa-f7184a80380e.png" width="250">
    <img src="https://user-images.githubusercontent.com/13795717/80310199-744abd00-8814-11ea-91e9-6135ae9a3d8c.png" width="250">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/13795717/80310155-33eb3f00-8814-11ea-98c7-f0fb68c00f33.png" width="250">
    <img src="https://user-images.githubusercontent.com/13795717/80310204-790f7100-8814-11ea-97ef-a98e6caadc46.png" width="250">
</div>

