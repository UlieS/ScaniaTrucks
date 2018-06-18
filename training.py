from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import evaluation


def train_knn(train, test, seed):
    np.random.seed(seed)

    # the labels
    x_columns = list(train)[2:]
    y_column = ['class']

    # create a classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=10, n_jobs=4)
    knn.fit(train[x_columns], train[y_column].values.ravel())

    pred_proba = knn.predict_proba(test[x_columns])
    pred = knn.predict(test[x_columns])

    for i in range(len(pred)):
        if(pred_proba[i][0] >= 0.1):
            pred[i] = 0

    return pred


def train_randomForest(train, test, seed, metrics, i):
    np.random.seed(seed)
    features = train.columns[2:]
    y = train['class']
    clf = RandomForestClassifier(criterion='gini',n_jobs=-1, bootstrap=False, max_depth=6, max_features=20, min_samples_leaf=17, min_samples_split=6, class_weight={1:50, 0:1})  # , random_state=0)
    clf.fit(train[features], y)
    pred = clf.predict(test[features])
    feature_importance=list(zip(train[features],clf.feature_importances_))
    return pred, feature_importance
