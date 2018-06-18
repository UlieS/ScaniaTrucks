import pandas as pd
import os
import data_cleaning
import training
import numpy as np
import evaluation
import plotting
from operator import itemgetter


# activate this if you want to clean the data
# df = data_cleaning.clean_data()

train = pd.read_csv(os.getcwd()+"/data/training_imputed.csv",
                    na_values="na", dtype=np.float)
train["class"]=train["class"].apply(lambda x: 1 if x == 0 else 0)

# test=data_cleaning.impute(test,saveAs="test_imputed.csv")


test = pd.read_csv(os.getcwd()+"/data/test_imputed.csv",
                   na_values="na", dtype=np.float)
test["class"]=test["class"].apply(lambda x: 1 if x == 0 else 0)
# train RandomForest and knn a certain amount of iterations and get evaluation metrics

metrics = [[0]*5, [0]*5]
r = list(range(1000))
np.random.shuffle(r)
r=r[:20]
plotdata=[]
for i in r:
    seed = i
    # train the knn classifier
    #pred_knn = training.train_knn(train, test)
    #metrics[1] = evaluation.createEvaluationMetrics(test["class"], pred_knn, metrics[1])
    # train the random forest classifier
    pred_rf, feat_imp= training.train_randomForest(train, test, seed, metrics,maxDepth)
    metrics[0] = evaluation.createEvaluationMetrics(test["class"], pred_rf, metrics[0])
    plotdata.append(evaluation.createPlottingMetrics(test["class"], pred_rf))

print(metrics)
plotting.LinePlot(plotdata)



'''
used for testing different parameters, feature numbers

#plotdata=[]    
feat_imp = (sorted(feat_imp,key=itemgetter(1)))
y=[x[0] for x in feat_imp[len(feat_imp)-17:]]
y[0]="class"
train_lessFeat = train[y]
test_lessFeat = test[y]

r = list(range(1000))
#np.random.shuffle(r)
r=r[:10]
for i in r:
    maxDepth += 1
    nrOfFeat += 5
    seed = i
    # train the knn classifier
    #pred_knn = training.train_knn(train, test)
    #metrics[1] = evaluation.createEvaluationMetrics(test["class"], pred_knn, metrics[1])
    # train the random forest classifier
    pred_rf, feat_imp= training.train_randomForest(train_lessFeat, test_lessFeat, seed, metrics,1)
    metrics[1] = evaluation.createEvaluationMetrics(test["class"], pred_rf, metrics[1])
    plotdata.append(evaluation.createPlottingMetrics(test["class"], pred_rf))


#plotting.featPlot(feat_imp)
#plotting.LinePlot(plotdata)


# metrics = [i/len(r) for i in metrics]
# evaluation.printMetrics(metrics)
print(metrics)
'''