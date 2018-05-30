import pandas as pd
import os
from imputer import Imputer


df = pd.read_csv(os.getcwd()+"/data/aps_failure_training_set.csv",
                 na_values="na", dtype=str)

# transform class label to neg=1 pos=0
df["class"] = df["class"].apply(lambda x: 1 if x == 'neg' else 0)
df = df.apply(pd.to_numeric)

# get percentage of NaN per column
missingValues = df.isnull().sum()/60000

# use kNN imputation to fill in missing values
impute = Imputer()
df = impute.knn(X=df, column='ab_000', k=3)
