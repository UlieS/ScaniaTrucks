
def train(train, test):

    # the labels
    x_columns = list(train)[1:171]
    y_column = ['class']

    # create a regressor
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(train[x_columns], train[y_column])

    predictions = knn.predict(test[x_columns])
