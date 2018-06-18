import pandas as pd
import numpy as np
import training
import sklearn.metrics as mt


def createEvaluationMetrics(X, pred, metrics):
    '''
    collects confusion matrix from each iteration, calculates relevant metrics 
    and stores them in a dictionary

    input: confusionmatrix[pd.DataFrame]
    output: dictionary containing all important values 
    test
    '''

    confMat = mt.confusion_matrix(X,pred)
    tn, fp, fn, tp = confMat.ravel()
    
    total = tn+fp+fn+tp
    # Total costs
    metrics[0] += fn*500+fp*10
    # Accuracy
    metrics[1] += (tp+tn)/total
    # Precision
    prec = tp/(tp+fp)
    metrics[2] += prec
    # Recall
    rec = tp/(tp+fn)
    metrics[3] += rec
    # F1 Score
    metrics[4] += 2*((rec*prec)/(rec+prec))
    print(metrics)
    print(confMat)
#    for metric,value in evals.items():
#        print(metric+ ": "+ str(value))
    # pred_prob=pd.DataFrame(clf.predict_proba(test[features]))
    #feature_importance=list(zip(train[features], clf.feature_importances_))
    return metrics

def printMetrics(metrics,times):
    # get average 
    avgMetrics = [i/times for i in metrics]
    metricNames = ["Total Costs","Accuracy","Precision","Recall", "F1 Measure"]
    print("Average evaluation measures after "+ str(times) +" iterations")
    for i in range(len(metricNames)):
        print(metricNames[i]+ ": " + str(avgMetrics[i])+ "\n")


def createPlottingMetrics(X,pred):
    confMat = mt.confusion_matrix(X,pred)
    tn, fp, fn, tp = confMat.ravel()
    metrics=[0]*5
    total = tn+fp+fn+tp
    # Total costs
    metrics[0] = fn*500+fp*10
    # Accuracy
    metrics[1] = (tp+tn)/total
    # Precision
    prec = tp/(tp+fp)
    metrics[2] = prec
    # Recall
    rec = tp/(tp+fn)
    metrics[3] = rec
    # F1 Score
    metrics[4] = 2*((rec*prec)/(rec+prec))
    print(metrics)
    return metrics
