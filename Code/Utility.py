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

def convertDataFrame(inputDataPath, colname, totalnum=0):
    '''
    convert inputdata (in inputDataPath) to Bert supported format
    inputDataPath: file path for the inputdata
    colname: column name contains category
    totalnum: desired total count after data balancing. If not set (0) or over 
             the input data length, it will be ignored. If it is within the data
             length, it will try to balance the data.
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
    
    #Prepare for data balancing
    dobalance = False
    if totalnum == 0:
        print("Total number is not set. Skipping data balancing")
    elif totalnum >= len(inputdata):
        print("Total number is too big. Skipping data balancing")
    else:
        dobalance = True
        required = totalnum
        catdict = {}
        for i in range(len(categories)):
            catdict[categories[i]] = len(inputdata[inputdata[colname] == categories[i]])
        #Sort in ascending order
        catdict = sorted(catdict.items(), key=lambda kv: kv[1])
        
    #Build output dataframe
    out = pd.DataFrame(columns = ['ID', 'Text'] + categories)
    
    if dobalance == True:
        for i in range(0, len(catdict)):
            curdata = inputdata[inputdata[colname] == catdict[i][0]]
            buf = pd.DataFrame(columns = out.columns)
            buf['ID'] = curdata['ID']
            buf['Text'] = curdata['Text']
            buf[categories] = [0]*len(categories)
            buf[catdict[i][0]] = 1
            average = int(required / (len(categories) - i))
            if len(buf) >= average:
                buf = buf.sample(average)
            out = out.append(buf)
            required = required - len(buf)
            print(catdict[i][0] + ":" + str(len(buf)))
    else:
        for i in range(0, len(categories)):
            curdata = inputdata[inputdata[colname] == categories[i]]
            buf = pd.DataFrame(columns = out.columns)
            buf['ID'] = curdata['ID']
            buf['Text'] = curdata['Text']
            buf[categories] = [0]*len(categories)
            buf[categories[i]] = 1
            out = out.append(buf)
            
    return(out.reset_index(drop=True))

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
    plt.title('Multi-class ROC Comparison')
    plt.legend(loc="lower right")
    plt.show()
    
    return([roc_auc, fpr, tpr])