# Weka CatBoost
# An implementation of CatBoost in Weka, using Weka PyScript
# For Capitol Technology University's Computer Science Lab

# 7/7/2021: COMPLETE, TESTED

# Parameters for Weka:
    # num_leaves (default 31)
    # learning_rate (default 0.03)
    # n_estimators (I utilized 1000)
    
#This is the code that can be utilized to run CatBoost with specefic parameters you want to provide within the Argument box within Weka

from wekapyscript import ArffToArgs
import catboost

# Train the model
def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    CB = catboost.CatBoostClassifier(num_leaves = args["num_leaves"], learning_rate = args["learning_rate"], n_estimators = args["n_estimators"])
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
