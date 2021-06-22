# Weka LightGBM
# An implementation of LightGBM in Weka, using Weka PyScript
# For Capitol Technology University's Computer Science Lab

# 6/21/2021: Incomplete, untested

# Parameters for Weka:
    # num_leaves (default 32)
    # learning_rate (default 0.05)
    # n_estimators (default 20)
    
# TODO: Add parameters and usage guide to readme

import lightgbm as lgb
from wekapyscript import ArffToArgs

# Train the model
# TODO: Test in Weka
def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    
    lgbm = lgb.LGBMRegressor(num_leaves = args["num_leaves"], learning_rate = args["learning_rate"], n_estimators = args["n_estimators"])
    lgbm.fit(x_train, y_train)
    return lgbm
    
# Model description
# TODO: Add args to return string
def describe(args, model):
    return "LightGBM"
    
# Test the model
# TODO: Test in Weka
def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()