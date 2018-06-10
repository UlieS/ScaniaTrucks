import pandas as pd
import os
import data_cleaning
import training
import numpy as np
import evaluation


# activate this if you want to clean the data
#df = data_cleaning.clean_data()

train = pd.read_csv(os.getcwd()+"/training_imputed_allFeatures.csv",
                    na_values="na", dtype=np.float64)
#print(train.describe())

# impute test set - also apply same data cleaning procedure as training set?
#test=data_cleaning.impute(test,saveAs="test_imputed.csv")

test = pd.read_csv(os.getcwd()+"/test_imputed.csv",
                   na_values="na", dtype=str)

# transform class label to neg=1 pos=0
#test["class"] = test["class"].apply(lambda x: 1 if x == 'neg' else 0)
test = test.apply(pd.to_numeric)

#test=data_cleaning.clean_data(test)

#training.train(train, test)

# train RandomForest certain amount of iterations and get evaluation metrics
metrics = [0]*4
r=list(range(10))
np.random.shuffle(r)
for i in r:
    seed=i
    metrics=training.randomForest(train,test,seed,metrics)

metrics = [i/len(r) for i in metrics]
evaluation.printMetrics(metrics)




