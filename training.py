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
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train[x_columns], train[y_column].values.ravel())

    predictions = knn.predict(test[x_columns])
    return predictions


def train_randomForest(train,test,i,metrics):
    np.random.seed(i)
    features=train.columns[2:]
    y=train['class']
    weights=y.apply(lambda x: 0.99 if x == 1 else 0.01)
    clf = RandomForestClassifier(n_jobs=2)#, random_state=0)
    clf.fit(train[features], y, sample_weight=weights)
    pred=clf.predict(test[features])
    
    confMat=pd.crosstab(test['class'], pred, rownames=['Actual class'], colnames=['Predicted class'])
    print(confMat)
    metrics=evaluation.getEvaluationMetrics(confMat,metrics)
    #pred_prob=pd.DataFrame(clf.predict_proba(test[features]))
    feature_importance=list(zip(train[features], clf.feature_importances_))
    return metrics, feature_importance

    
