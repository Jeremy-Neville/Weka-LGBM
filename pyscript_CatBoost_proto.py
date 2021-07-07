# Weka CatBoost
# An implementation of CatBoost in Weka, using Weka PyScript
# For Capitol Technology University's Computer Science Lab

# 7/7/2021: COMPLETE, TESTED

#This is the prototype code that can be utilized to run CatBoost within Weka witout any parameters being provided into the arguments section within Weka.
from wekapyscript import ArffToArgs
import catboost

def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    CB = catboost.CatBoostClassifier()
    CB.fit(x_train, y_train)
    return CB



def describe(args, model):
    return "catboost is running"

def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()
