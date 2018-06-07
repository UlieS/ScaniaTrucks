import pandas as pd
import os
import data_cleaning
import training
import numpy as np

# activate this if you want to clean the data
#df = data_cleaning.clean_data()

train = pd.read_csv(os.getcwd()+"/training_imputed.csv",
                    na_values="na", dtype=np.float64)

# impute test set - also apply same data cleaning procedure as training set?
#test=data_cleaning.impute(test,saveAs="test_imputed.csv")

test = pd.read_csv(os.getcwd()+"/test_imputed.csv",
                   na_values="na", dtype=str)

# transform class label to neg=1 pos=0
test["class"] = test["class"].apply(lambda x: 1 if x == 'neg' else 0)
test = test.apply(pd.to_numeric)



#training.train(train, test)

#feat=train.columns[1:]
#clf=training.randomForest(train[feat],test)
#print(clf)