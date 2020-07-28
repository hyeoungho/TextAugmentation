# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:40:19 2020

@author: hybae
"""

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

def convertDataFrame(inputDataPath, colname):
    '''
    convert inputdata (in inputDataPath) to Bert supported format
    inputDataPath: file path for the inputdata
    colname: column name contains category
    '''
    #Read input data
    try:
        inputdata = pd.read_csv(inputDataPath, encoding='latin1')
    except BaseException as e:
        raise Exception("Could not read inputdata:" + inputDataPath + "Error: " + str(e))
    
    try:
        categories = list(np.unique(inputdata[colname]))
    except BaseException as e:
        raise Exception("Could not find column name: " + colname)
    
    #Build output dataframe
    out = pd.DataFrame(columns = ['ID', 'Text'] + categories)
    out['ID'] = inputdata['ID']
    out['Text'] = inputdata['Text']
    for i in range(0, len(inputdata)):
        cat = [0]*len(categories)
        cat[categories.index(inputdata.loc[i, colname])] = 1
        out.loc[i, categories[0]:] = cat
    
    return(out)
    
def ROCAnalysis(y_true, y_pred):
    if all(y_true.columns == y_pred.columns) != True:
        raise Exception("Column does not match between GT and Prediction data")
    classes = y_true.columns
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in classes:
        fpr[i], tpr[i], _ = roc_curve(y_true.loc[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])    