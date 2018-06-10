import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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
        

#def Scatterplot