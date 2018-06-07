from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

np.random.seed(0)


def train_knn(train, test):

    # the labels
    x_columns = list(train)[1:171]
    y_column = ['class']

    # create a regressor
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train[x_columns], train[y_column].values.ravel())

    predictions = knn.predict(test[x_columns])
    return predictions


def train_randomForest(train, test):
    features = train.columns[1:]
    # y=pd.factorize(train['class'])[0]
    y = train['class']
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(train[features], y)
    pred = clf.predict(test[features])
    return pred
