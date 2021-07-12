# Weka - LGBM & CatBoost PyScript

## Introduction
Within here you will see documentation of how to implement LGBM and CatBoost within Weka through PyScript. You will see the differences between the prototype and the finished product. You will also be given an idea as to how to run the py scripts with and without arguments within Weka. Furthermore, you will be able to see the results provided by the new updated ProtoType with the current Dataset I have available to test these pyscripts with. 

This is done to ensure that any users that want to implement LGBM or CatBoost within Weka can do so with a PyScript either with or without specific parameters. Overall this is made to ensure that there is documentation regarding the implementation of LGBM and CatBoost in Weka

**NOTE** : The dataset you may utilize this pyscript may be very different from my current dataset so please remember the results will not always be the same and may require you to do some experimentation on your data set prior to running the pyscript to attain results that can be deemed the best for the data you provided to the model.

# How to install CatBoost and LGBM

Before we go into detail regarding the prototypes and updated prototypes, It is important to ensure that LightGBM, and CatBoost are installed onto your python. To ensure that, we need to ensure our Python has [pip](https://phoenixnap.com/kb/install-pip-windows) installed. After ensuring that PIP is installed, we will then need to install LightGBM, and CatBoost.

**NOTE** : installing pip on your device is very different if you are using Linux. Provided here is a link regarding installing software on your Linux device through the [Command Line](https://opensource.com/article/18/8/how-install-software-linux-command-line) 

## Opening / Finding Command Prompt

Before we go deeper into the installation of LGBM and CatBoost, it is important to provide information regarding opening command prompt on your computer. Provided below are links regarding opening your command prompt/terminal on your device.

[MAC](https://support.apple.com/guide/terminal/open-or-quit-terminal-apd5265185d-f365-44cb-8b09-71a064a42125/mac)

[WINDOWS](https://www.ionos.com/digitalguide/server/tools/open-command-prompt/)

[LINUX](https://ubuntu.com/tutorials/command-line-for-beginners#3-opening-a-terminal)



## Installing LGBM(LightGBM)

When it comes to installing LGBM, all you need to do is run the pip command displayed below in your command prompt.

```
pip install lightgbm
```

However, it is possible that one does not have wheel installed which could hinder the installation process, so it is important to install wheel as well just to ensure you can install lightgbm without any issues. To download wheel, all you need to do is run the pip command displayed below in the command prompt.

```
pip install wheel
```

## Installing CatBoost

Once you have installed LGBM, all that's needed is to download CatBoost to ensure that your pyscript can run without any issues. To download CatBoost, all that's needed to be done is to run the pip command displayed below in the command prompt.

```
pip install catboost
```
 
# Arguments/Parameters within Weka
To ensure the py script with parameters works, you need to ensure that within Weka and where it says arguments you include the parameters with the specific values you desire. This will be reiterated within each prototype and final version of the PyScript for CatBoost and LGBM.

EX:

```
#The following line is how you would need to input the values within the arguments textbox within Weka, to ensure that the py script with num_leaves,learning_rate, and n_estimators can run without error.

num_leaves=32;learning_rate=0.05;n_estimators=20
```
![arguments](https://user-images.githubusercontent.com/49813790/124758556-af5a7d00-defc-11eb-89d0-b7033177cc8a.PNG)


# LGBM ProtoType
What we will now focus on is the original prototype I created for my LGBM py script I utilized to attain my original results from the prototype. It is important to note that the code for the prototype will only have a few differences compared to the finished product. 

## LGBM ProtoType Code
The code provided below is the code utilized for the prototype. As you can see, the LGBMClassifier does not utilize any parameters and is instead relying upon what can be inferred to be default parameters. What we are currently unsure of is if these default parameters are good parameters or are insufficient in allowing us to acquire a good understanding of how effective LGBM is with this data set.
```
# Weka LightGBM ProtoType
# An implementation of LightGBM in Weka, using Weka PyScript
# For Capitol Technology University's Computer Science Lab

# 7/7/2021: COMPLETE, TESTED

#This is the prototype code that you can utilize to simply run LGBM without any parameters given by you the user
from wekapyscript import ArffToArgs
import lightgbm as lgb

# Train the model
def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    lgbm = lgb.LGBMClassifier()
    lgbm = lgbm.fit(x_train, y_train)
    return lgbm

# Model description
# TODO: Add args to return string
def describe(args, model):
    return "LightGBM is running"

# Test the model
# TODO: Test in Weka
def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()

```

## Running the lGBM Prototype
![LGBMPROTO](https://user-images.githubusercontent.com/49813790/124766522-019f9c00-df05-11eb-87c3-10b98d736c36.PNG)


As shown above, since you are not utilizing any parameters / Arguments, you do not need to do anything except indicating to Weka that the py script you want to utilize is the LGBM Prototype PyScript.


## LGBM ProtoType Results
As you can see, what is shown below are the results from running the prototype with a 10 fold cross-validation. These results however are not final due to the fact I am far from fully understanding how to acquire the best results utilizing parameters and utilizing said parameters in Weka, especially if there is more than one parameter.

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



# The New LGBM ProtoType
I was able to figure out the small issues that were preventing me from acquiring the results I wanted. What will be displayed below will be the results given by the code which had some alteration which will be detailed.

## The Alteration Made Within The Code, and how to apply it within your prototype code
The biggest alteration I made when it comes to the code was to change the LGBMRegressor statement to an LGBM classifier statement. This alteration was made to ensure I can run it within Weka as a classifier with the default parameters. All you need to do within your prototype code is replace the previous declaration of lgbm with the following line of code.
```
lgbm= lgb.LGBMClassifier(num_leaves = args["num_leaves"], learning_rate = args["learning_rate"], n_estimators = args["n_estimators"])
```

## Running The LightGBM PyScript Within Weka With Arguments
![arguments](https://user-images.githubusercontent.com/49813790/124758556-af5a7d00-defc-11eb-89d0-b7033177cc8a.PNG)

To ensure that the py script is running without issue it is important that num_leaves=32, learning_rate=0.05, and n_estimators=20. These are the default parameters you should apply when first running this py script within weka to ensure the py script runs without any issues.


## The Results From The Alteration
I was able to put the default parameters within Weka, which then led to the following results to be provided by this model. It becomes clear that this alteration within the code allowed the classifier to provide results that were different from when I ran the classifier without any parameters. This doesn't mean that the results will differ greatly from the ones that have provided the best results however it shows that progress is being made.

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
What we will now focus on is the original prototype I created for the CatBoost py script I utilized to attain my original results from the prototype. It is important to note that the code for the prototype will only have a few differences compared to the finished product. It also is important to note that the code for the prototype is very similar to the LGBM prototype code.

## CatBoost ProtoType Code
As you can see below, the code for the CatBoost Prototype is very similar to the LGBMK Prototype code however as you can see this one as well can't be fully trusted due to it utilizing the default parameters that we are unsure if are good parameters or are the worst parameters to provide the model. Overall this code will be revised upon alongside the LGBM code to fully ensure that the results given are the best results possible.
```
# Weka CatBoost ProtoType
# An implementation of CatBoost in Weka, using Weka PyScript
# For Capitol Technology University's Computer Science Lab

# 7/7/2021: COMPLETE, TESTED

#This is the prototype code that can be utilized to run CatBoost within Weka without any parameters being provided into the arguments section within Weka.
from wekapyscript import ArffToArgs
import catboost

# Train the model
def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    CB = catboost.CatBoostClassifier()
    CB.fit(x_train, y_train)
    return CB

# Model description
# TODO: Add args to return string
def describe(args, model):
    return "catboost is running"

# Test the model
# TODO: Test in Weka
def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()
```


## Running the CatBoost ProtoType
![CATBOOSTPROTO](https://user-images.githubusercontent.com/49813790/124769222-4b898180-df07-11eb-8a59-d96649cf4634.PNG)

As reiterated before, the CatBoost prototype allows you to run CatBoost without any arguments. So to run it within Weka all you need to do is have the same screen like the one shown above since you are not providing any arguments to the model.

## CatBoost ProtoType Results
As shown below, the results below are somewhat similar to those of the LGBM prototype. However, as stated before this result is not the final result because we haven't applied enough parameters to safely state that our results are the best results given by CatBoost.

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
Since I was able to figure out to effectively create the prototype for LGBM, I began working on updating the prototype for CatBoost. This led to me to create an updated py script which was able to provide me some results that would be deemed beneficial in ensuring that we can provide specific parameters that can provide us better results.

## The update within the CatBoost PyScript
As displayed below, the only change made within the code was the additional parameters that were included within the declaration of the CatBoost Classifier. Utilizing the CatBoost prototype code provided, the only change you need to do is replace your previous declaration of CB with the one displayed in the line of code below.
```
CB = catboost.CatBoostClassifier(num_leaves = args["num_leaves"], learning_rate = args["learning_rate"], n_estimators = args["n_estimators"])
```
## The parameters I utilized within Weka for CatBoost
Within weka, I placed the following values as the arguments that would be passed when I ran my newly revised CatBoost Pyscript ProtoType. I began searching as to what values will be accepted as default parameters to be utilized when running the py script.

```
num_leaves=31;learning_rate=0.03;n_estimators=1000
```
It would be recommended to decrease the value of n_estimators to around 10-20 to ensure the model is truly efficient.

## Running the CatBoost Pyscript within Weka
![CatBoost_argument](https://user-images.githubusercontent.com/49813790/124760273-9226ae00-defe-11eb-9bda-169530dfb0ae.PNG)

To ensure that the updated CatBoost Pyscript works with arguments, it is important to run it within Weka. As displayed above I would recommend setting 
num_leaves=31, learning_rate=0.03 and n_estimators=20. Shown above is how exactly you should see your py script before running it within Weka through a 10 fold cross-validation.

## The Results From The Updated CatBoost Pyscript ProtoType 

>=== Classifier model (full training set) ===

>catboost is running

>Time taken to build model: 5.59 seconds

>=== Stratified cross-validation ===

>=== Summary ===

>Correctly Classified Instances      124203               98.5949 %

>Incorrectly Classified Instances      1770                1.4051 %

>Kappa statistic                          0.9717

>Mean absolute error                      0.0917

>Root mean squared error                  0.1283

>Relative absolute error                 18.4227 %

>Root relative squared error             25.7265 %

>Total Number of Instances           125973     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.995    0.024    0.979      0.995    0.987      0.972    0.999     0.999     normal
                 0.976    0.005    0.994      0.976    0.985      0.972    0.999     0.999     anomaly
    Weighted Avg.    0.986    0.015    0.986      0.986    0.986      0.972    0.999     0.999     

>=== Confusion Matrix ===

     a     b   <-- classified as
    66980   363 |     a = normal
    1407 57223 |     b = anomaly


>=== Re-evaluation on test set ===

>User supplied test set

>Relation:     KDDTest

>Instances:     unknown (yet). Reading incrementally

>Attributes:   42

>=== Summary ===

>Correctly Classified Instances       18126               80.4028 %

>Incorrectly Classified Instances      4418               19.5972 %

>Kappa statistic                          0.6181

>Mean absolute error                      0.2338

>Root mean squared error                  0.3723

>Total Number of Instances            22544     

>=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.963    0.316    0.697      0.963    0.809      0.652    0.958     0.958     normal
                 0.684    0.037    0.961      0.684    0.799      0.652    0.958     0.949     anomaly
    Weighted Avg.    0.804    0.157    0.847      0.804    0.803      0.652    0.958     0.953     

>=== Confusion Matrix ===

    a    b   <-- classified as
    9351  360 |    a = normal
     4058 8775 |    b = anomaly
     
     
# BONUS: returning the parameters you gave to your model

What we have established right now is the backbone of ensuring this model can be trained. However, to ensure users don't have to worry that the model their training is being trained with the correct parameters it is important to return such information. Currently, within both the updated prototypes as we will name them, they only state either LGBM or CatBoost is running. This is something we will need to change

## The revised code

The only aspect you need to change is within the describe function. This is something that can be done within the two updated prototypes, and is a very small change. There is one small issue, that being that the learning_rate is only displayed as 0 however once a resolution is found regarding that small issue, that will be available within this README. This code is not something that is mandatory but is recommended to ensure you can see the parameters for these specific pyscripts.

## The code implemented in Updated LGBM Prototype
To ensure the code is understandable, and to pinpoint a small difference. These two sections are added to ensure no user is confused by what small aspect of the text is changed within the code of the two describe functions displayed. The following code is the code that should be implemented within your updated LGBM Prototype.

```
def describe(args, model):
    text = "LightGBM with %i leaves, " % args["num_leaves"]
    text_two = "a learning rate of %i, and " % args["learning_rate"]
    text_three= "%i estimators." % args["n_estimators"]
    return (text+text_two+text_three)
```

## The code implemented in Updated CatBoost Prototype
The code displayed before is the code that should be implemented into the Updated CatBoost Prototype. Remember that this code is optional however we are making this distinction to ensure no user copies the previous code into their updated CatBoost prototype or other pyscripts that rely upon arguments without noticing the small distinction found within the text variable.

    # This is optional code, however, it is good to provide this within this file to ensure all users can understand the template for displaying multiple parameters.
    
    #text = "CatBoost with %i leaves, " % args["num_leaves"]
    #text_two = "a learning rate of %i, and " % args["learning_rate"]
    #text_three= "%i estimators." % args["n_estimators"]
    #return (text+text_two+text_three)
