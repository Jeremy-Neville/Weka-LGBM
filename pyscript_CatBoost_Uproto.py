# Parameters for Weka:
    # num_leaves (default 31)
    # learning_rate (default 0.03)
    # n_estimators (I utilized 1000)
  
from wekapyscript import ArffToArgs
import catboost

def train(args):
    x_train = args["X_train"]
    y_train = args["y_train"]
    CB = catboost.CatBoostClassifier(num_leaves = args["num_leaves"], learning_rate = args["learning_rate"], n_estimators = args["n_estimators"])
    CB.fit(x_train, y_train)
    return CB



def describe(args, model):
    return "catboost is running"

def test(args, model):
    X_test = args["X_test"]
    return model.predict_proba(X_test).tolist()
