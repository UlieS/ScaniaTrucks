import pandas as pd
import os
import data_cleaning
import training

# activate this if you want to clean the data
#  df = data_cleaning.clean_data()

train = pd.read_csv(os.getcwd()+"/training_imputed.csv",
                    na_values="na", dtype=str)

test = pd.read_csv(os.getcwd()+"/data/aps_failure_test_set.csv",
                   na_values="na", dtype=str)
# transform class label to neg=1 pos=0
test["class"] = test["class"].apply(lambda x: 1 if x == 'neg' else 0)
test = test.apply(pd.to_numeric)

training.train(train, test)
