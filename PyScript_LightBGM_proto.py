# Weka LightGBM
# An implementation of LightGBM in Weka, using Weka PyScript
# For Capitol Technology University's Computer Science Lab

# 7/7/2021: COMPLETE, TESTED

#This is the prototype code that you can utilize to simply run LGBM without any parameters given by you the user
from wekapyscript import ArffToArgs
import lightgbm as lgb

def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    lgbm = lgb.LGBMClassifier()
    lgbm = lgbm.fit(x_train, y_train)
    return lgbm


def describe(args, model):
    return "LightGBM is running"


def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()
