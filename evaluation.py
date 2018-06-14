import pandas as pd
import numpy as np
import training



    
def getEvaluationMetrics(confMat,metrics):
    '''
    collects confusion matrix from each iteration, calculates relevant metrics 
    and stores them in a dictionary
    
    input: confusionmatrix[pd.DataFrame]
    output: dictionary containing all important values 
    
    '''
    
    tp = confMat.iloc[0,0]
    fp = confMat.iloc[1,0]
    tn = confMat.iloc[1,1]
    fn = confMat.iloc[0,1]
    total = tp+tn+fp+fn
    
    # Total costs
    metrics[0] += fn*500+fp*10
    # Accuracy
    metrics[1] += (tp+tn)/total
    # Precision
    metrics[2] += tp/(tp+fp)
    # Recall
    metrics[3] += tp/(tp+fn)
    
#    print(confMat)
#    for metric,value in evals.items():
#        print(metric+ ": "+ str(value))
    #pred_prob=pd.DataFrame(clf.predict_proba(test[features]))
    #feature_importance=list(zip(train[features], clf.feature_importances_))
    return metrics

def printMetrics(metrics,times):
    # get average 
    avgMetrics = [i/times for i in metrics]
    metricNames = ["Total Costs","Accuracy","Precision","Recall"]
    print("Average evaluation measures after "+ str(times) +" iterations")
    for i in range(len(metricNames)):
        print(metricNames[i]+ ": " + str(avgMetrics[i])+ "\n")