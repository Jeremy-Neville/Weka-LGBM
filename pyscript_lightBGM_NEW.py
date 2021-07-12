# Weka LightGBM
# An implementation of LightGBM in Weka, using Weka PyScript
# For Capitol Technology University's Computer Science Lab

# 7/7/2021: COMPLETE, TESTED

# Parameters for Weka:
    # num_leaves (default 32)
    # learning_rate (default 0.05)
    # n_estimators (default 20)
    
#This is the code that can be utilized to run LightGBM with specefic parameters you want to provide within the Argument box within Weka

import lightgbm as lgb
from wekapyscript import ArffToArgs

# Train the model
def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    lgbm= lgb.LGBMClassifier(num_leaves = args["num_leaves"], learning_rate = args["learning_rate"], n_estimators = args["n_estimators"])
    lgbm.fit(x_train, y_train)
    return lgbm
    
# Model description
# TODO: Add args to return string
def describe(args, model):
    
    return "LightGBM is running"


    # As displayed within the readme, there is a way to display the parameters inputted by the user. This is the way I was able to to display all three parameters without any issues
    # This is optional code, however it is good to provide this within this file to ensure all users can understand the template for displaying multiple parameters.
    
    #text = "LightGBM with %i leaves, " % args["num_leaves"]
    #text_two = "a learning rate of %i, and " % args["learning_rate"]
    #text_three= "%i estimators." % args["n_estimators"]
    #return (text+text_two+text_three)

   
    
# Test the model
# TODO: Test in Weka
def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()
