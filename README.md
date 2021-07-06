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


## LGBM ProtoType Results
As you can see, what is shown below are the results from running the prototype with a 10 fold corss validation.These results however are not final due to the fact I am far from fully understanding how to acquire the best results utilizing parameters and utilizing said parameters in Weka, especially if there are more than one parameter.

> === Classifier model (full training set) ===

>lighgbm is running

>Time taken to build model: 13.75 seconds

>=== Stratified cross-validation ===
>=== Summary ===

>Correctly Classified Instances      125850               99.9024 %

>Incorrectly Classified Instances       123                0.0976 %

>Kappa statistic                          0.998 

>Mean absolute error                      0.0018

>Root mean squared error                  0.0278

>Relative absolute error                  0.3629 %

>Root relative squared error              5.565  %

>Total Number of Instances           125973     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.999    0.001    0.999      0.999    0.999      0.998    1.000     1.000     normal
                 0.999    0.001    0.999      0.999    0.999      0.998    1.000     1.000     anomaly
    Weighted Avg.    0.999    0.001    0.999      0.999    0.999      0.998    1.000     1.000     

>=== Confusion Matrix ===

    a     b   <-- classified as
    67296    47 |     a = normal
    76 58554 |     b = anomaly
    
>
>=== Re-evaluation on test set ===

>User supplied test set

>Relation:     KDDTest

>Instances:     unknown (yet). Reading incrementally

>Attributes:   42
>=== Summary ===

>Correctly Classified Instances       17808               78.9922 %

>Incorrectly Classified Instances      4736               21.0078 %

>Kappa statistic                          0.5928

>Mean absolute error                      0.2087

>Root mean squared error                  0.4369

>Total Number of Instances            22544     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.970    0.346    0.679      0.970    0.799      0.635    0.965     0.962     normal
                 0.654    0.030    0.967      0.654    0.780      0.635    0.965     0.971     anomaly
    Weighted Avg.    0.790    0.166    0.843      0.790    0.788      0.635    0.965     0.967     


>=== Confusion Matrix ===

    a    b   <-- classified as
    9421  290 |    a = normal
    4446 8387 |    b = anomaly 


# LGBM Results With Different Parameters
For some reason before I was unable to have the three parameters within my weka for some reason which I was unable to figure out until recently as to why that is occuring. So just in case I was able to make some modifications on the code provided to make sure it works with each singular parameter and documented them.

## 32 Leaves Results
>=== Classifier model (full training set) ===
>LightGBM

>Time taken to build model: 5.51 seconds


>=== Stratified cross-validation ===
>=== Summary ===

>Correctly Classified Instances      125857               99.9079 %

>Incorrectly Classified Instances       116                0.0921 %

>Kappa statistic                          0.9981

>Mean absolute error                      0.0017

>Root mean squared error                  0.0274

>Relative absolute error                  0.3509 %

>Root relative squared error              5.4833 %

>Total Number of Instances           125973     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.999    0.001    0.999      0.999    0.999      0.998    1.000     1.000     normal
                 0.999    0.001    0.999      0.999    0.999      0.998    1.000     1.000     anomaly
    Weighted Avg.    0.999    0.001    0.999      0.999    0.999      0.998    1.000     1.000     

>=== Confusion Matrix ===

     a     b   <-- classified as
    67297    46 |     a = normal
    70 58560 |     b = anomaly

>=== Re-evaluation on test set ===

>User supplied test set

>Relation:     KDDTest

>Instances:     unknown (yet). Reading incrementally

>Attributes:   42

>=== Summary ===

>Correctly Classified Instances       17694               78.4865 %

>Incorrectly Classified Instances      4850               21.5135 %

>Kappa statistic                          0.5836

>Mean absolute error                      0.2113

>Root mean squared error                  0.4407

>Total Number of Instances            22544   
  
>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.971    0.356    0.674      0.971    0.795      0.628    0.967     0.964     normal
                 0.644    0.029    0.967      0.644    0.773      0.628    0.967     0.972     anomaly
    Weighted Avg.    0.785    0.170    0.841      0.785    0.783      0.628    0.967     0.969     

>=== Confusion Matrix ===

    a    b   <-- classified as
    9427  284 |    a = normal
    4566 8267 |    b = anomaly


## Learning_Rate=0.05 Results
>=== Classifier model (full training set) ===
>LightGBM

>Time taken to build model: 6.35 seconds

>=== Stratified cross-validation ===
>=== Summary ===

>Correctly Classified Instances      125765               99.8349 %

>Incorrectly Classified Instances       208                0.1651 %

>Kappa statistic                          0.9967

>Mean absolute error                      0.0072

>Root mean squared error                  0.0366

>Relative absolute error                  1.4443 %

>Root relative squared error              7.3402 %

>Total Number of Instances           125973     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.999    0.002    0.998      0.999    0.998      0.997    1.000     1.000     normal
                 0.998    0.001    0.999      0.998    0.998      0.997    1.000     1.000     anomaly
    Weighted Avg.    0.998    0.002    0.998      0.998    0.998      0.997    1.000     1.000     

>=== Confusion Matrix ===

     a     b   <-- classified as
    67261    82 |     a = normal
    126 58504 |     b = anomaly

>=== Re-evaluation on test set ===

>User supplied test set

>Relation:     KDDTest

>Instances:     unknown (yet). Reading incrementally

>Attributes:   42

>=== Summary ===

>Correctly Classified Instances       17563               77.9054 %

>Incorrectly Classified Instances      4981               22.0946 %

>Kappa statistic                          0.5731

>Mean absolute error                      0.2084

>Root mean squared error                  0.4254

>Total Number of Instances            22544

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.971    0.366    0.667      0.971    0.791      0.619    0.961     0.958     normal
                 0.634    0.029    0.966      0.634    0.766      0.619    0.961     0.961     anomaly
    Weighted Avg.    0.779    0.174    0.838      0.779    0.777      0.619    0.961     0.960     

>=== Confusion Matrix ===

    a    b   <-- classified as
    9427  284 |    a = normal
    4697 8136 |    b = anomaly



## n_estimators=20 Results
>=== Classifier model (full training set) ===
>LightGBM

>Time taken to build model: 10.01 seconds

>=== Stratified cross-validation ===

>=== Summary ===

>Correctly Classified Instances      125605               99.7079 %

>Incorrectly Classified Instances       368                0.2921 %

>Kappa statistic                          0.9941

>Mean absolute error                      0.0709

>Root mean squared error                  0.0831

>Relative absolute error                 14.2462 %

>Root relative squared error             16.6569 %

>Total Number of Instances           125973     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.998    0.004    0.997      0.998    0.997      0.994    0.999     0.999     normal
                 0.996    0.002    0.997      0.996    0.997      0.994    0.999     0.999     anomaly
    Weighted Avg.    0.997    0.003    0.997      0.997    0.997      0.994    0.999     0.999     

>=== Confusion Matrix ===
>
>     a     b   <-- classified as
> 67190   153 |     a = normal
>   215 58415 |     b = anomaly
>
>=== Re-evaluation on test set ===
>
>User supplied test set

>Relation:     KDDTest

>Instances:     unknown (yet). Reading incrementally

>Attributes:   42

>=== Summary ===

>Correctly Classified Instances       17743               78.7039 %

>Incorrectly Classified Instances      4801               21.2961 %

>Kappa statistic                          0.5877

>Mean absolute error                      0.2549

>Root mean squared error                  0.4193

>Total Number of Instances            22544     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.971    0.352    0.676      0.971    0.797      0.631    0.910     0.831     normal
                 0.648    0.029    0.967      0.648    0.776      0.631    0.910     0.921     anomaly
>Weighted Avg.    0.787    0.168    0.842      0.787    0.785      0.631    0.910     0.882     

>=== Confusion Matrix ===

    a    b   <-- classified as
    9431  280 |    a = normal
    4521 8312 |    b = anomaly


# The New LGBM ProtoType
I was able to figure out the small issues that were preventing me from acquiring the results I wanted. What will be doisplayed below will be the results given by the code which had some alteration which will be detailed.

## The Alteration Made Within The Code
The biggest alteration I made when it comes to the code was change the LGBMRegressor statement to an LGBM classifier statement. This alteration was made to ensure I can run it within Weka as a classifier with the default parameters.
```
lgbm= lgb.LGBMClassifier(num_leaves = args["num_leaves"], learning_rate = args["learning_rate"], n_estimators = args["n_estimators"])
```

## The Results From The Alteration
I was able to put the default parameters within Weka, which then led to the following results to be provided by this model. It becomes clear that this alteration within the code allowed the classifer to provide results that were different from when I ran the classifier without any parameters. This doesn;t mean that the results will differ greatly from the ones that have provided the best results however it shows that progress is being made.

>=== Classifier model (full training set) ===
>LightGBM

>Time taken to build model: 6.82 seconds

>=== Stratified cross-validation ===

>=== Summary ===

>Correctly Classified Instances      125537               99.6539 %

>Incorrectly Classified Instances       436                0.3461 %

>Kappa statistic                          0.993 

>Mean absolute error                      0.1856

>Root mean squared error                  0.1892

>Relative absolute error                 37.3049 %

>Root relative squared error             37.9377 %

>Total Number of Instances           125973 
    
>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.998    0.005    0.996      0.998    0.997      0.993    0.999     0.999     normal
                 0.995    0.002    0.997      0.995    0.996      0.993    0.999     0.999     anomaly
   Weighted Avg.    0.997    0.004    0.997      0.997    0.997      0.993    0.999     0.999     

>=== Confusion Matrix ===

     a     b   <-- classified as
    67185   158 |     a = normal
    278 58352 |     b = anomaly

>=== Re-evaluation on test set ===

>User supplied test set

>Relation:     KDDTest

>Instances:     unknown (yet). Reading incrementally

>Attributes:   42

>=== Summary ===

>
>Correctly Classified Instances       17867               79.2539 %

>Incorrectly Classified Instances      4677               20.7461 %

>Kappa statistic                          0.5978

>Mean absolute error                      0.3222

>Root mean squared error                  0.4067

>Total Number of Instances            22544 
    
>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.973    0.344    0.682      0.973    0.802      0.640    0.906     0.828     normal
                 0.656    0.027    0.969      0.656    0.783      0.640    0.906     0.918     anomaly
    Weighted Avg.    0.793    0.164    0.845      0.793    0.791      0.640    0.906     0.879     

>=== Confusion Matrix ===

    a    b   <-- classified as
    9445  266 |    a = normal
    4411 8422 |    b = anomaly




 
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

## CatBoost ProtoType Results
As shown below, the results below are somewhat similar to those of the LGBM prototype. However as stated before this result is not the final result due to the fact that we haven't applied enough parameters to safely state that our results are the best results given by CatBoost.
>=== Classifier model (full training set) ===

>catboost is running


>Time taken to build model: 211.83 seconds


>=== Stratified cross-validation ===

>=== Summary ===

>Correctly Classified Instances      125766               99.8357 %

>Incorrectly Classified Instances       207                0.1643 %

>Kappa statistic                          0.9967

>Mean absolute error                      0.0042

>Root mean squared error                  0.0362

>Relative absolute error                  0.8471 %

>Root relative squared error              7.2488 %

>Total Number of Instances           125973     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.999    0.003    0.998      0.999    0.998      0.997    1.000     1.000     normal
                 0.997    0.001    0.999      0.997    0.998      0.997    1.000     1.000     anomaly
    Weighted Avg.    0.998    0.002    0.998      0.998    0.998      0.997    1.000     1.000     

=== Confusion Matrix ===

     a     b   <-- classified as
    67284    59 |     a = normal
    148 58482 |     b = anomaly

>=== Re-evaluation on test set ===

>User supplied test set

>Relation:     KDDTest

>Instances:     unknown (yet). Reading incrementally

>Attributes:   42

>=== Summary ===

>Correctly Classified Instances       17965               79.6886 %

>Incorrectly Classified Instances      4579               20.3114 %

>Kappa statistic                          0.6056

>Mean absolute error                      0.1991

>Root mean squared error                  0.4163

>Total Number of Instances            22544  
   
>
>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.971    0.335    0.687      0.971    0.805      0.645    0.970     0.971     normal
                 0.665    0.029    0.968      0.665    0.789      0.645    0.970     0.967     anomaly
    Weighted Avg.    0.797    0.161    0.847      0.797    0.795      0.645    0.970     0.969     

>=== Confusion Matrix ===

    a    b   <-- classified as
    9426  285 |    a = normal
    4294 8539 |    b = anomaly

# The New CatBoost Prototype
Since I was able to figure out to effictively create the prototype for LGBM, I began working on updating the prototype for CatBoost. This led to me to create an updated pyscript which was able to provide me some results that would be deemed benificial in ensuring that we can provide specefic parameters that can provide us better results.

## The update within the CatBoost PyScript
As displayed below, the only change made within the code was the additional parameters that were included within the declaration of the CatBoost Classifier. I also commented in the parameters I utilized within Weka  as to what to provide a better idea as to what each parameter within the code was given when I was placing in values for each parameter within weka.
```
# Parameters for Weka:
    # num_leaves (default 31)
    # learning_rate (default 0.03)
    # n_estimators (I utilized 1000)
catboost.CatBoostClassifier(num_leaves = args["num_leaves"], learning_rate = args["learning_rate"], n_estimators = args["n_estimators"])
```
## The parameters I utilized within Weka
Within weka, I placed the following values as the arguments that would be passed when I ran my newly revised CatBoost Pyscript ProtoType. I began searching as to what values will be acceptable as default parameters to be utilized when running the pyscript.

```
num_leaves=31;learning_rate=0.03;n_estimators=1000
```
## The Results From The Updated CatBoost Pyscript ProtoType

>=== Classifier model (full training set) ===

>catboost is running

>Time taken to build model: 103.55 seconds

>=== Stratified cross-validation ===

>=== Summary ===

>Correctly Classified Instances      125753               99.8254 %

>Incorrectly Classified Instances       220                0.1746 %

>Kappa statistic                          0.9965

>Mean absolute error                      0.0044

>Root mean squared error                  0.0371

>Relative absolute error                  0.8929 %

>Root relative squared error              7.4355 %

>Total Number of Instances           125973     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.999    0.003    0.998      0.999    0.998      0.996    1.000     1.000     normal
                 0.997    0.001    0.999      0.997    0.998      0.996    1.000     1.000     anomaly
>Weighted Avg.    0.998    0.002    0.998      0.998    0.998      0.996    1.000     1.000     

>=== Confusion Matrix ===

     a     b   <-- classified as
> 67278    65 |     a = normal
>   155 58475 |     b = anomaly


>=== Re-evaluation on test set ===

>User supplied test set

>Relation:     KDDTest

>Instances:     unknown (yet). Reading incrementally

>Attributes:   42

>=== Summary ===

>Correctly Classified Instances       18199               80.7266 %

>Incorrectly Classified Instances      4345               19.2734 %

>Kappa statistic                          0.6247

>Mean absolute error                      0.195 

>Root mean squared error                  0.4113

>Total Number of Instances            22544     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.971    0.316    0.699      0.971    0.813      0.661    0.966     0.967     normal
                 0.684    0.029    0.969      0.684    0.802      0.661    0.966     0.966     anomaly
    Weighted Avg.    0.807    0.153    0.852      0.807    0.806      0.661    0.966     0.966     

>=== Confusion Matrix ===

    a    b   <-- classified as
    9426  285 |    a = normal
    4060 8773 |    b = anomaly
