import pandas as pd
import os

df=pd.read_csv(os.getcwd()+"/aps_failure_training_set.csv",dtype=str)
#print(df['ab_000'])


#check amounts of na and 0 per column
for column in df:
    print(df[column].isnull().count())