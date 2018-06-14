from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import evaluation


def train_knn(train, test):

    # the labels
    x_columns = list(train)[1:171]
    y_column = ['class']

    # create a classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    knn.fit(train[x_columns], train[y_column].values.ravel())

    pred_proba = knn.predict_proba(test[x_columns])
    pred = knn.predict(test[x_columns])

    for i in range(len(pred)):
        if(pred_proba[i][0] >= 0.2):
            pred[i] = 0

    return pred


def train_randomForest(train, test, seed, metrics):
    np.random.seed(seed)
    features = train.columns[2:]
    y = train['class']
    weights = y.apply(lambda x: 0.95 if x == 1 else 0.05)
    clf = RandomForestClassifier(n_jobs=2)  # , random_state=0)
    clf.fit(train[features], y, sample_weight=weights)
    pred = clf.predict(test[features])
    #feature_importance=list(zip(train[features],clf.feature_importances_))
    return pred
