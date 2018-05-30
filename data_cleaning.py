import pandas as pd
import os
from imputer import Imputer


df=pd.read_csv(os.getcwd()+"/data/aps_failure_training_set.csv",na_values="na",dtype=str)

# transform class label to neg=1 pos=0 
df["class"]=df["class"].apply(lambda x: 1 if x=='neg' else 0)
df=df.apply(pd.to_numeric)

# get percentage of NaN per column
missingValues=df.isnull().sum()/60000

# use kNN imputation to fill in missing values 
df2=[]
impute= Imputer()
col=0
for col in range(df.shape[1]):
   result= impute.knn(X=df, column=col, k=3)
   df.iloc[:,col]=result[:,col]
   print(col)
   
df.tofile('imputed.csv',sep=',',format='%10.5f')



