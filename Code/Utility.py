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
import tensorflow as tf
import tensorflow_hub as hub


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
    if totalnum == 0 or totalnum < 0:
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

def balancingData(inputdata, totalnum):
    '''
    Data balancing for inputdata with already in Bert Compatible form
    '''
    if totalnum == 0 or totalnum < 0:
        print("Total number is not set. Skipping data balancing")
        return
    elif totalnum >= len(inputdata):
        print("Total number is too big. Skipping data balancing")
        return
    #Build output dataframe
    out = pd.DataFrame(columns = inputdata.columns)
    categories = list(inputdata.columns[2:])
    required = totalnum
    catdict = {}
    for i in range(len(categories)):
        catdict[categories[i]] = len(inputdata[inputdata[categories[i]] == 1])
        #Sort in ascending order
        catdict = sorted(catdict.items(), key=lambda kv: kv[1])
    for i in range(0, len(catdict)):
        curdata = inputdata[inputdata[catdict[i][0]] == 1]
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
    
    return(out.reset_index(drop=True))
     
def is_ascii(s):
    '''
    Simply checks if the input sentence (s) contains non-ascii characters. If so, returns False. Otherwise, returns True
    #Arguments
    # s: input sentence (string)
    '''
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
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
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

def embed_useT(module = None):
    if module == None:
        module = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


class textAnalyzer:
    def __init__(self, data, embed_fn):
        '''
        Assumption on data columns: 'Text'
        '''
        
        data = data.reset_index(drop=True)
        self.doc = data['Text']
        self.embed_fn = embed_fn
        
        try:
            print("***Vectorize using USE***")
            self.vectorizeData()
            print("***Converting to 2D vector***")
            self.convert2D()       
            
        except BaseException as e:
            print("Failed to vectorize:" + str(e))
    
    def get_features(self, texts):
        if type(texts) is str:
            texts = [texts]
        return self.embed_fn(texts)

    def vectorizeData(self):
        if (len(self.doc) == 0):
            print("Nothing to vectorize")
            return
        self.vectors = []
        for i in range(0, len(self.doc)):
            self.vectors.append(self.get_features(self.doc[i]))
        self.vectors = np.array(self.vectors)
        self.vectors = self.vectors[:,0,:]
        self.mean = np.mean(self.vectors, axis=0)
        self.std = np.std(self.vectors)
    
    def clustering(self, clusternum):
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=clusternum, random_state=0).fit(self.vectors)
        self.clusterlabels=self.kmeans.labels_
    
    def convert2D(self):
        from sklearn.manifold import TSNE
        RS = 23
        self.vectors2d = TSNE(random_state=RS).fit_transform(self.vectors)
    
    def compareDataset(self, datasets, labels):
        import seaborn as sns
        '''
        Draw scatter plot from different dataset in datasets
        '''
        datasetsnum = len(datasets)
        palette = np.array(sns.color_palette("hls", datasetsnum))
        
        f = plt.figure(figsize=(32, 32))
        ax = plt.subplot(aspect='equal')
        for i in range(0, len(datasets)):
            sc = ax.scatter(datasets[i][:,0], datasets[i][:,1], lw=0, s=120, 
                            label = labels[i], c=palette[[i]*len(datasets[i])])
        ax.legend()
        ax.grid(True)
        plt.show()
    
    def plotselfcluster(self):
        self.clustering(18)
        self.scatter(self.vectors2d, self.clusterlabels)
    
    def scatter(self, x, colors):
        import seaborn as sns
        import matplotlib.patheffects as PathEffects
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", 18))
    
        # We create a scatter plot.
        f = plt.figure(figsize=(32, 32))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                        c=palette[colors.astype(np.int)])
        #plt.xlim(-25, 25)
        #plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')
        
        # We add the labels for each cluster.
        txts = []
        for i in range(18):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=50)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    
        return f, ax, sc, txts

def compareDataset(dataset, labels):
    import seaborn as sns
    '''
    Draw scatter plot from different dataset in datasets
    '''
    datasetsnum = len(dataset)
    palette = np.array(sns.color_palette("hls", datasetsnum))
    
    f = plt.figure(figsize=(32, 32))
    ax = plt.subplot(aspect='equal')
    for i in range(0, len(dataset)):
        sc = ax.scatter(dataset[i][:,0], dataset[i][:,1], lw=0, s=120, label=labels[i],
                        c=palette[[i]*len(dataset[i])])
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.legend()
    ax.grid(True)
    plt.show()

def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

def makedatasets(inputdatapath, embed_fn):
    '''
    Making dataset that can be used for the scatter plot (compareDataset)
    
    '''
    inputdata = pd.read_csv(inputdatapath, encoding = 'latin1')
    categories = inputdata.columns[2:]
    dataset = []
    labels = []
    for i in range(0, len(categories)):
        od = inputdata[inputdata[categories[i]] == 1]
        dc = textAnalyzer(od, embed_fn)
        dataset.append(dc)
        labels.append(categories[i])
    return(dataset, labels)

def showDistrtibution(vectors, mvector, drawhist=True):
    dist = []
    mag2 = np.linalg.norm(mvector)
    if (not mag2):
        return(np.array([0]*len(vectors)))
    for i in range(len(vectors)):
        mag1 = np.linalg.norm(vectors[i])
        if (not mag1):
            dist.append(0)
        dist.append(np.dot(vectors[i], mvector) / (mag1 * mag2))
    if drawhist == True:
        _ = plt.hist(dist, bins = 'auto')
    dist = np.array(dist)
    mean = np.mean(dist)
    print(mean)
    return dist
    
    