import pandas as pd
import os
import data_cleaning
import training
import numpy as np
import evaluation


# activate this if you want to clean the data
# df = data_cleaning.clean_data()

train = pd.read_csv(os.getcwd()+"/training_imputed_allFeatures.csv",
                    na_values="na", dtype=np.float64)

# test=data_cleaning.impute(test,saveAs="test_imputed.csv")

test = pd.read_csv(os.getcwd()+"/test_imputed.csv",
                   na_values="na", dtype=str)

# transform class label to neg=1 pos=0
test = test.apply(pd.to_numeric)

# train RandomForest and knn a certain amount of iterations and get evaluation metrics

metrics = [[0]*4, [0]*4]
r = list(range(1))
np.random.shuffle(r)
for i in r:
    seed = i
    # train the knn classifier
    pred_knn = training.train_knn(train, test)
    metrics[1] = evaluation.createEvaluationMetrics(test, pred_knn, metrics[1])
    # train the random forest classifier
    pred_rf = training.train_randomForest(train, test, seed, metrics)
    metrics[0] = evaluation.createEvaluationMetrics(test, pred_rf, metrics[0])


# metrics = [i/len(r) for i in metrics]
# evaluation.printMetrics(metrics)
print(metrics)
