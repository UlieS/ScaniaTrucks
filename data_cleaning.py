import pandas as pd
import os
from imputer import Imputer
import matplotlib.pyplot as plt
import numpy as np


def clean_data():
    df = pd.read_csv(os.getcwd()+"/data/aps_failure_training_set.csv",
    na_values="na", dtype=str)


    # transform class label to neg=1 pos=0
    df["class"] = df["class"].apply(lambda x: 1 if x == 'neg' else 0)
    df = df.apply(pd.to_numeric)


    # get percentage of NaN per column
    missingValues = df.isnull().sum()/60000 

    '''
    # generate boxplot of missing values
    fig, ax = plt.subplots()
    ax.boxplot(np.array(missingValues))
    ax.set_title("Percentage of missing values per feature")
    fig.savefig(os.getcwd()+"/figures/boxplot.png")
    '''

    # take out features with more than 60% missing values based on visual analysis of boxplot
    colsToDrop=missingValues.where(missingValues >0.6)
    df=df.drop(columns=(colsToDrop[colsToDrop.notnull()].index))

    
    # potentially take out samples that have more than x amount of unknown values
    #missingrows=[]
    #for row in range(len(df)):  
    #    missingrows.append(df.iloc[row].isnull().sum())
    
#    corrMatrix.where(corrMatrix>0.95).count().sort_values()   # see how many features correlate highly with another one
#    corrMatrix.where(corrMatrix>0.95).count().sum-len(corrMatrix)   # 258-163=95 features are highly correlated

    corrMatrix=df.corr(method='pearson')
    corrMatrix=corrMatrix.where(corrMatrix>0.95) 
    highlyCorrelated={}
    for col in corrMatrix:
        highlyCorrelated[col]=corrMatrix.index[corrMatrix[col].notnull()].tolist()
   
    # use kNN imputation to fill in missing values
    '''
    impute = Imputer()
    for col in range(df.shape[1]):
        result = impute.knn(X=df, column=col, k=3)
        df.iloc[:, col] = result[:, col]
        print(col)

    df.to_csv('training_imputed.csv', sep=',')
    '''
    return df, highlyCorrelated

df, corr=clean_data()
