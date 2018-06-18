import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab
from matplotlib.font_manager import FontProperties

def Boxplot(data, title="", savedName="boxplot.png", save=True):
    '''
    generate boxplot of missing values
    
    input: data [list], optional: title [string], savedName [string], save [bool]
    output: boxplot.png in ./figures
    
    '''
    fig, ax = plt.subplots()
    ax.boxplot(np.array(data))
    ax.set_title(title)
    if save:
        fig.savefig(os.getcwd()+"/figures/"+savedName)
        

def LinePlot(data):
    metricNames = ["Total Costs","Accuracy","Precision","Recall", "F1 Measure"]
    #x_axes=[5+5*i for i in range(20)]
    x_axes=[i+1 for i in range(20)]
    data=np.array(data)
    m,n=data.shape

    plt.subplot(2,1,1)
    
    
    plt.ylabel('Total Cost')
    plt.plot(x_axes,data[:,0], '.-')
   

    plt.subplot(2,1,2)

    for col in range(1,n):
        plt.plot(x_axes,data[:,col], '.-', label=metricNames[col])
    
    #pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.legend(loc='best', shadow=True)   
    plt.xlabel('Maximum depth per tree')
    plt.show()



def featPlot(data):

    x=[x[0]for x in data]
    i=0
    for el in range(len(x)):
        i += 1
        if el==2:
            continue
        x[el] = str(i)+"_"+x[el]

    y=[x[1] for x in data]
    plt.bar(x,y)
    plt.show()
