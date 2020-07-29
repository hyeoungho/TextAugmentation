# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:40:19 2020

@author: hybae
"""

import pandas as pd
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
import re, sys, traceback

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

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def sanitizedata(inputdata):
    '''
    Cleanup inputdata so that it can be used for training
    #Assumption on the inputdata
     It should contain 'Text' as its document column
    '''
    dropind = []
    try:
        for i in range(0, len(inputdata)) :
            doc = inputdata.loc[i, 'Text']
            if is_ascii(doc) != True:
                dropind.append(i)
            else:
                #Do some cleaning
                doc = re.sub('\[(.*?)\]', '', doc).strip()
                if len(doc) < 10:
                    #Too short
                    dropind.append(i)
                else:
                    inputdata.loc[i, 'Text'] = doc
        inputdata = inputdata.drop(dropind)
        return(inputdata.reset_index(drop=True))
    except BaseException as e:
        print("*** print_tb:")
        traceback.print_exc(file=sys.stdout)
        raise Exception("Failed to sanitize data:" + str(e))

def ROCAnalysis(y_true, y_pred):
    '''
    Run ROC analysis and draw plot    
    #Arguments
    # y_true: DataFrame containing the ground truth 
    # y_pred: DataFrame containing prediction
    '''
    if all(y_true.columns == y_pred.columns) != True:
        raise Exception("Column does not match between GT and Prediction data")
    classes = y_true.columns
    n_classes = len(classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #Should remove any data type for roc_curve
    _y_true = y_true.to_numpy().astype(None)
    _y_pred = y_pred.to_numpy().astype(None)
    for i in range(0, n_classes):
        fpr[i], tpr[i], _ = roc_curve(_y_true[:, i], _y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(_y_true.ravel(), _y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
    return([roc_auc, fpr, tpr])