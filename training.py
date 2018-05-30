
def train(df):

    # the labels
    x_columns = list(df)[1:171]
    y_column = ['class']

    # create a regressor
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(df[x_columns], df[y_column])
