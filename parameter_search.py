from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import evaluation
import os
from scipy.stats import randint as sp_randint
from time import time
import sklearn.metrics as mt

def loss_function(y, pred):
    tn, fp, fn, tp = mt.confusion_matrix(y,pred, labels=[1,0]).ravel()
    mat=mt.confusion_matrix(y,pred)
    return (fp*10+fn*500)*-1

def parameterSearch():
    loss=mt.make_scorer(loss_function)

    train = pd.read_csv(os.getcwd()+"/data/training_imputed.csv",
                        na_values="na", dtype=np.float)
    train["class"]=train["class"].apply(lambda x: 1 if x == 0 else 0)
    features = train.columns[1:]
    test = pd.read_csv(os.getcwd()+"/data/test_imputed.csv",
                    na_values="na", dtype=np.float)
    test["class"]=test["class"].apply(lambda x: 1 if x == 0 else 0)
    test=test[features]
    train=train.append(test[features], ignore_index=True)
    X=train[features]
    y=train["class"]
    # build a classifier
    clf = RandomForestClassifier(n_estimators=10)


    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")


    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [6,10, 3, None],
                "max_features": sp_randint(40, 60),
                "min_samples_split": sp_randint(2, 20),
                "min_samples_leaf": sp_randint(15, 40)}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                    n_iter=n_iter_search, scoring=loss)

    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
        " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)


parameterSearch()