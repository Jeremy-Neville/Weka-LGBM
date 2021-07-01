# Weka-LGBM

## Introduction
Within here you will see a documentation of my progress in implementing LGBM and CatBoost within Weka through PyScript. You will see the differences between the prototype and the finished product.

# LGBM ProtoType
What we will now focus on is the original prototype I created for my LGBM pyscript I utilized to attain my orignal results from the prototype. It is important to note that the code for the prototype will only have a few differences compared to the finished product. 

## LGBM ProtoType Code
The code below is the code utilized for the prototype. As you can see, the LGBMClassifier does not utilize any parameters and is instead relying upon what can be inferred to be default parameters. What we are currently unsure of is if these default parameters are good parameters or are insufficient in allowing us to acquire a good understanding of how effective LGBM is with this data set.
```
#I keep the two statements that were imported in the RandomForest Classifier pyscript just in case
from wekapyscript import ArffToArgs
import lightgbm

def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    rf = lightgbm.LGBMClassifier()
    rf = rf.fit(x_train, y_train)
    return rf


def describe(args, model):
    return "lighgbm is running"


def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()
```


## LGBM ProtoType results
As you can see, what is shown below are the results from running the prototype with a 10 fold corss validation.These results however are not final due to the fact I am far from fully understanding how to acquire the best results utilizing parameters and utilizing said parameters in Weka, especially if there are more than one parameter.

> === Classifier model (full training set) ===
>
>lighgbm is running
>
>Time taken to build model: 13.75 seconds
>
>=== Stratified cross-validation ===
>=== Summary ===
>
>Correctly Classified Instances      125850               99.9024 %
>Incorrectly Classified Instances       123                0.0976 %
>Kappa statistic                          0.998 
>Mean absolute error                      0.0018
>Root mean squared error                  0.0278
>Relative absolute error                  0.3629 %
>Root relative squared error              5.565  %
>Total Number of Instances           125973     
>
>=== Detailed Accuracy By Class ===
>
>                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
>                 0.999    0.001    0.999      0.999    0.999      0.998    1.000     1.000     normal
>                 0.999    0.001    0.999      0.999    0.999      0.998    1.000     1.000     anomaly
>Weighted Avg.    0.999    0.001    0.999      0.999    0.999      0.998    1.000     1.000     
>
>=== Confusion Matrix ===
>
>     a     b   <-- classified as
> 67296    47 |     a = normal
>    76 58554 |     b = anomaly
>
>
>=== Re-evaluation on test set ===
>
>User supplied test set
>Relation:     KDDTest
>Instances:     unknown (yet). Reading incrementally
>Attributes:   42
>
>=== Summary ===
>
>Correctly Classified Instances       17808               78.9922 %
>Incorrectly Classified Instances      4736               21.0078 %
>Kappa statistic                          0.5928
>Mean absolute error                      0.2087
>Root mean squared error                  0.4369
>Total Number of Instances            22544     
>
>=== Detailed Accuracy By Class ===
>
>                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
>                 0.970    0.346    0.679      0.970    0.799      0.635    0.965     0.962     normal
>                 0.654    0.030    0.967      0.654    0.780      0.635    0.965     0.971     anomaly
>Weighted Avg.    0.790    0.166    0.843      0.790    0.788      0.635    0.965     0.967     
>
>=== Confusion Matrix ===
>
>    a    b   <-- classified as
> 9421  290 |    a = normal
> 4446 8387 |    b = anomaly



 
# CatBoost ProtoType
What we will now focus on is the original prototype I created for my CatBoost pyscript I utilized to attain my orignal results from the prototype. It is important to note that the code for the prototype will only have a few differences compared to the finished product. It also is important to note that the code for the prototype is very similar to the LGBM prototype code.

## CatBoost ProtoType Code
As you can see below, the code for the CatBoost Prototype is very similar to the LGBMK Prototype code however as you can see this one as well can't be fully trusted due to it utilizing the default parameters that we are unsure if are good parameters or are the worst parameters to provide the model. Overall this code will be revised upon alongside the LGBM code to fully ensure that the results given are the best results possible.
```
from wekapyscript import ArffToArgs
import lightgbm
import catboost

def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    rf = catboost.CatBoostClassifier()
    rf.fit(x_train, y_train)
    return rf



def describe(args, model):
    return "catboost is running"

def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()
```


