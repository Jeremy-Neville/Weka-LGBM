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
