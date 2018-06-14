import pandas as pd
import os
import data_cleaning
import training
import numpy as np
import evaluation


# activate this if you want to clean the data
#df = data_cleaning.clean_data()

train = pd.read_csv(os.getcwd()+"/data/training_imputed.csv",
                    na_values="na", dtype=np.float64)

# impute test set - also apply same data cleaning procedure as training set?
# test=data_cleaning.impute(test,saveAs="test_imputed.csv")

test = pd.read_csv(os.getcwd()+"/data/test_imputed.csv",
                   na_values="na", dtype=str)

test = test.apply(pd.to_numeric)

#training.train_knn(train, test)

metrics = [0]*4
iterationTimes=100
r=list(range(1000))
np.random.shuffle(r)
r=r[:1]
for i in r:
    seed=i
    metrics, feat_imp=training.train_randomForest(train,test,seed,metrics)

evaluation.printMetrics(metrics,iterationTimes)




