import pandas as pd
import os
from imputer import Imputer
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter


def genBoxplot(data, title, savedName, save=True):
    '''
    generate boxplot of missing values
    
    input: data [list], title [string] 
    output: boxplot.png in /figures
    
    '''
    fig, ax = plt.subplots()
    ax.boxplot(np.array(data))
    ax.set_title(title)
    if save:
        fig.savefig(os.getcwd()+"/figures/"+savedName+".png")
    
    
def mergeFeatures(corrMatrix,threshold):
    '''
    analyzes the correlation between the features and compares whether highly correlated 
    features correlate highly with the same set of other features
    returns the features to be merged 
    
    input: correlation matrix [pd.DataFrame], threshold of minimum correlation [int] 
    output: list of features to be merged [list of lists]
    
    '''
    corrMatrix=corrMatrix.where(corrMatrix>threshold) 
    
#    corrMatrix.where(corrMatrix>0.95).count().sort_values()   # see how many features correlate highly with another one, 
                                                                # cd_000 dooes not even correlate with itself?
#    corrMatrix.where(corrMatrix>0.95).count().sum-len(corrMatrix)   # 258-163=95 features are highly correlated

    highlyCorrelated={}
    for col in corrMatrix:
        if (corrMatrix.index[corrMatrix[col].notnull()].size>1):
            highlyCorrelated[col]=corrMatrix.index[corrMatrix[col].notnull()].tolist()
   
    highlyCorrelated=sorted(highlyCorrelated.items(),key=itemgetter(1))
    result=[]
    key=1
    while key<len(highlyCorrelated):
        toBeMerged=[]
        toBeMerged.append(highlyCorrelated[key-1][0])
        while key<len(highlyCorrelated) and highlyCorrelated[key][1]==highlyCorrelated[key-1][1]:
            toBeMerged.append(highlyCorrelated[key][0])
            key+=1
        if len(toBeMerged)>1:
            result.append(toBeMerged)
        key+=1
    return result


def clean_data():
    df = pd.read_csv(os.getcwd()+"/data/aps_failure_training_set.csv",
    na_values="na", dtype=str)


    # transform class label to neg=1 pos=0
    df["class"] = df["class"].apply(lambda x: 1 if x == 'neg' else 0)
    df = df.apply(pd.to_numeric)


    # get percentage of NaN per column
    missingValues = df.isnull().sum()/60000 
    title="Percentage of missing values per feature"
    saveAs="boxplot1.png"
    genBoxplot(missingValues, title,saveAs)


    # take out features with more than 60% missing values based on visual analysis of boxplot
    colsToDrop=missingValues.where(missingValues >0.6)
    df=df.drop(columns=(colsToDrop[colsToDrop.notnull()].index))

    corrMatrix=df.corr(method='pearson')
    toBeMerged=mergeFeatures(corrMatrix, 0.95)
    
    # keep the first feature of all sets of highly correlated features and drop the rest
    list(map(lambda x: x.pop(0),toBeMerged))
    colsToDrop = [item for sublist in toBeMerged for item in sublist]
    df=df.drop(columns=colsToDrop)
        
    
    # potentially take out samples that have more than x amount of unknown values
    #missingrows=[]
    #for row in range(len(df)):  
    #    missingrows.append(df.iloc[row].isnull().sum())
    
    
    
    # use kNN imputation to fill in missing values
    '''
    impute = Imputer()
    for col in range(df.shape[1]):
        result = impute.knn(X=df, column=col, k=3)
        df.iloc[:, col] = result[:, col]
        print(col)

    df.to_csv('training_imputed.csv', sep=',')
    '''
    
    
    return df

df =clean_data()
