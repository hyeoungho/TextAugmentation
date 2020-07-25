# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:54:45 2020

@author: hybae
"""
import tensorflow as tf
import tensorflow_hub as hub
import nltk
from nltk.corpus import stopwords, wordnet
import numpy as np
import pandas as pd
import pickle
import os
import spacy
import re
import sys, traceback

class TextAugmentationClass:
    def __init__(self, outputpath, thval=0.8, applistpath=""):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        if os.path.isdir(outputpath) != True:
            os.mkdir(outputpath)
        self.outputpath = outputpath
        self.variablepath = outputpath + r'\variable.pckl'
        self.thval = thval
        #self.maxsyn = maxsyn
        self.embed_fn = self.embed_useT(module_url)
        self.data = pd.DataFrame(columns=['Text', 'Category', 'Keywords', 'HasKeywords'])
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = stopwords.words('english')
        
        try:
            if os.path.isfile(applistpath):
                self.applist = list(open(applistpath, encoding='latin1'))
                #cleanup
                for app in self.applist:
                    if self.is_ascii(app) != True:
                        self.applist.remove(app)
        except BaseException as e:
            print("Applist file open has failed: " + str(e))
    
    def caldatadist(self, inputdata):
        self.catlist = inputdata['Category'].str.get_dummies(sep=';').columns
        self.datadist = []
        self.weightlist = []
        total = len(inputdata)
        for i in range(0, len(self.catlist)):
            count = len(inputdata[inputdata['Category'].str.contains(self.catlist[i])])
            self.datadist.append(count)
            self.weightlist.append(total/(count + 1))
        m = min(self.weightlist)
        self.weightlist[:] = [x /m for x in self.weightlist]
        return
    
    def mergetext(newcolname, col_list, dt):
        if (len(col_list) < 1):
            raise Exception("col_list should not be empty")
        
        dt[newcolname] = dt[col_list[0]]
        if (len(col_list) > 1):
            for i in range(1, len(col_list)):
                dt[newcolname] = dt[newcolname] + ". " + dt[col_list[i]]
            
        return(dt)
    
    def storeData(self):
        try:
            if len(self.data) > 0:
                self.data.to_csv(self.outputpath, encoding="latin1", index=False)
                print("Successfully saved the data. Length:" + len(self.data))
            else:
                print("Nothing to save")
        except BaseException as e:
            print("Save the data has failed:" + str(e))
            return
        
    def is_ascii(self, s):
        return all(ord(c) < 128 for c in s)
    
    def embed_useT(self, module):
        with tf.Graph().as_default():
            sentences = tf.placeholder(tf.string)
            embed = hub.Module(module)
            embeddings = embed(sentences)
            session = tf.train.MonitoredSession()
        return lambda x: session.run(embeddings, {sentences: x})

    def find_synonym(self, w):
         synonyms = []
         for syn in wordnet.synsets(w): 
                for l in syn.lemmas(): 
                    synonyms.append(l.name())
         #synonyms = np.unique(synonyms)
         synonyms = pd.unique(synonyms)
         return synonyms
     
    def get_features(self, texts):
        if type(texts) is str:
            texts = [texts]
        return self.embed_fn(texts)
        #with tf.Session() as sess:
        #    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        #    return sess.run(embed(texts))
    
    def cosine_similarity(self, v1, v2):
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        if (not mag1) or (not mag2):
            return 0
        return np.dot(v1, v2) / (mag1 * mag2)
    
    def test_similarity(self, text1, text2):
        vec1 = self.get_features(text1)[0]
        vec2 = self.get_features(text2)[0]
        return self.cosine_similarity(vec1, vec2)
    
    def maxsynonym(self, words, postags, nertags, category):
        vbnum = 0
        maxsynnum = 0
        for i in range(0,len(words)):
            if np.sum(np.isin(self.stop_words, words[i])) == 0 and postags[i] == 'VERB' and nertags[i] == 'NK':
                vbnum = vbnum + 1
        if vbnum > 0:
            #print(str(vbnum))
            maxsynnum = 1000**(1/vbnum)
            if maxsynnum < 1:
                maxsynnum = 1
            else:
                maxsynnum = int(np.round(maxsynnum))
        #Apply weight
        _weight = 1
        category = category.split(';')
        try:
            if len(self.weightlist) > 0 :
                for cat in category:
                    if cat in self.catlist:
                        weight = self.weightlist[self.catlist.index(cat)]
                        if (weight > _weight):
                            _weight = weight
                        else:
                            weight = _weight
        except BaseException as e:
            #do nothing
            weight = 1
        return maxsynnum*weight
    
    def NERTagging(self, sent_tockenized, appnames) :
        #To deal with ';' delimited appnames
        appnames = appnames.split(';')
        appnames = np.unique(appnames)
        appnames = np.delete(appnames, np.where([x == '' for x in appnames]))
        n_tagged = []
        for i in range(0, len(appnames)):
            appnames_tockenized = nltk.word_tokenize(appnames[i])
            if i == 0 :
                for j in range(0, len(sent_tockenized)) :
                    n_tagged.append("NK")
            for j in range(0, len(appnames_tockenized)) :
                for k in range(0, len(sent_tockenized)) :
                    if (appnames_tockenized[j].lower() == sent_tockenized[k].lower()) :
                        if (k == 0 or (k > 0 and n_tagged[k - 1] == "NK")) :
                            n_tagged[k] = "SK"
                        else :
                            n_tagged[k] = "CK"
        
        #n_tagged = " ".join(n_tagged)
        return(n_tagged)
    
    def AugmentText(self, inputpath, outputpath, resume, savetodisk):
        #Assume that below columns to exist
        #Text, Keywords
        
        if resume == False:
            self.ind = 0
            #Read Data
            try:
                inputData = pd.read_csv(inputpath, encoding="latin1", keep_default_na=False)
            except BaseException as e:
                print("inputpath is invalid:" + str(e))
                return
            
            #Check column names:
            try:
                col_list = list(inputData.columns)
                if (('Text' not in col_list) or ('Keywords' not in col_list)):
                    raise Exception("Input data should have following columns: Text, Keywords")
                
                #Check if 'Category' column exists:
                if ('Category' not in col_list):
                    inputData['Category'] = ""
            except BaseException as e:
                print("Input data has problem:" + str(e))
            
            #Remove non-ascii rows:
            try:
                dropind = []
                for i in range(0, len(inputData)):
                    if self.is_ascii(inputData.loc[i, 'Text']) != True:
                        dropind.append(i)
                dropind = np.unique(dropind)
                inputData = inputData.drop(dropind)
                inputData.index = pd.RangeIndex(len(inputData.index))
                self.inputData = inputData
            except BaseException as e:
                print("InputData cleaning has failed:" + str(e))
                return
            
            #Calculate weight to cope with data imbalance
            self.caldatadist(inputData)
        else:
            #Load from the variable path
            if os.path.isdir(outputpath) != True:
                print("outputpath is invalid")
                return
            
            variablepath = outputpath + r'\variable.pckl'
            if os.path.isfile(variablepath) != True:
                print("variable to restore the state does not exist")
                return
            
            self = pickle.load(variablepath)
        #Augment text:
        try:
            for doc in self.nlp.pipe(self.inputData["Text"][self.ind:]):
                augres = pd.DataFrame(columns=['Text', 'Category', 'Keywords', 'HasKeywords'])
                #NERTag
                keywords = self.inputData.loc[self.ind, "Keywords"].strip()
                category = self.inputData.loc[self.ind, "Category"]
                if len(keywords) > 0:
                    nertags = self.NERTagging([token.text for token in doc], keywords)
                    haskey = 'TRUE'
                else:
                    #nertags = ' '.join(['NK']*len([token.text for token in doc]))
                    nertags = ['NK']*len([token.text for token in doc])
                    haskey = 'FALSE'
                #Replace title with lemmatized one
                x = ' '.join([token.text for token in doc])
                #snertags = ' '.join([str(token.ent_type) for token in doc])
                #postags = ' '.join([token.pos_ for token in doc])
                postags = [token.pos_ for token in doc]
                sentlist = self.text_augmentation(x, postags, nertags, category)
                #Add to the data
                sentlist.append(x)
                augres['Text'] = sentlist
                augres['Category'] = category
                augres['Keywords'] = keywords
                augres['HasKeywords'] = haskey
                self.data.append(augres)
                self.ind = self.ind + 1
                if (haskey == 'TRUE' and ('AppFunc' in category or 'AppInst' in category or 'Crash' in category)):
                    augres = self.AppNameAugmentation(augres, 5)
                
                #Store the state
                if (self.ind > 1 and self.ind % 10000 == 0):
                    print("Current progress: " + str(self.ind) + "/" + str(len(self.inputData)))
                    pickle.dump(self)
        except BaseException as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("Text augmentation has failed:" + str(e))
            print("*** print_tb:")
            traceback.print_exc(file=sys.stdout)
            return
        
        #Store the data
        try:
            if savetodisk == True:
                self.storeData()
        except BaseException as e:
            print("Data save has failed:" + str(e))
            return
    
    #Text augmentation function
    def text_augmentation(self, x, postags, nertags, category):
        words = x.split()
        synnum = self.maxsynonym(words, postags, nertags, category)
        if synnum == 0:
            print("text_augmentation: no candidate words")
            return []
        else:
            orgsent = x
            similarsent = []
            candidates = pd.unique(self.textaugmentation(x, postags, nertags, synnum, 0))
            #if (len(candidates) > self.maxsyn):
            #    candidates = candidates[:self.maxsyn]
            print("text_augmentation: " + str(len(candidates)) + " candidates")
            
            for i in range(0, len(candidates)):
                newsent = str(candidates[i])
                similarity = self.test_similarity(orgsent, newsent) 
                if (similarity > self.thval and similarity < 1):
                    #Should not include itself
                    similarsent.append(newsent)
            print("text_augmentation: " + str(len(similarsent)) + " similar sentences")
            return similarsent
    
    #Recursive text augmentation function
    def textaugmentation(self, x, postags, nertags, synnum, ind):
        '''This function is used to generate rows in the dataframe'''
        auglist = []
        words = x.split()
        for i in range(ind, len(words)):
            #print(str(i))
            if np.sum(np.isin(self.stop_words, words[i])) == 0 and postags[i] == 'VERB' and nertags[i] == 'NK':
                #Find synonym
                synonyms = []
                for syn in wordnet.synsets(words[i]): 
                    for l in syn.lemmas(): 
                        synonyms.append(l.name())
                synonyms = pd.unique(synonyms)
                synonyms = synonyms[:synnum]
                if len(synonyms) == 0:
                    synonyms = [words[i]]            
                elif np.sum(np.isin(synonyms, words[i])) == 0:
                    #print(str(i) + ": " + str(len(synonyms)) + ": " + words[i])
                    synonyms = np.insert(np.unique(synonyms), 0, words[i])
                    synonyms = synonyms[:synnum]
                
                if len(synonyms) > synnum:
                    synonyms = synonyms[:synnum]
                for j in range(0, len(synonyms)):
                    #Generate new sentence
                    newx = x
                    newwords = words
                    newwords[i] = synonyms[j]
                    newx = " ".join(newwords)
                    if (i < len(words) - 1) :
                        auglist.extend([newx] + self.textaugmentation(newx, postags, nertags, synnum, i + 1))
                    else :
                        auglist.extend([newx])
                break
        return auglist
    
    def AppNameAugmentation(self, rows, num):
        import random
        out = pd.DataFrame(columns = rows.columns)
        for i in range(0, len(rows)):
            appnames = rows.loc[i, 'Keywords'].split(';')
            while True:
                ind = random.randrange(0, len(appnames))
                oldapp = appnames[ind]
                if oldapp.lower() in rows.loc[i, 'Text'].lower():
                    break
            randomselect = []
            for i in range(0, num):
                while True:
                    newapp = random.choice(self.applist)
                    if newapp not in appnames and newapp not in randomselect:
                        randomselect.append(newapp)
                        appnames[ind] = newapp
                        newappnames = ';'.join(appnames)
                        cand = rows.loc[i,]
                        cand['Text'] = cand['Text'].replace(oldapp, newapp)
                        cand['Keywords'] = newappnames
                        out.append(cand, ignore_index=True)
                        break
            print('App name replacement added ' + str(len(out)) + ' sentences')
            out.append(rows, ignore_index=True)
        return
        
if __name__ == "__main__": 
    import sys
    #Argument list
    #inputpath, outputpath, applistpath, thval, resume, savetodisk
    #Ex: TextAugmenClass.py "c:\work\input" "c:\work\output" 0.9 100 False True
    if (len(sys.argv) != 7):
        print("System argument should be 7")
    else:
        try:
            inputpath = sys.argv[1]
            outputpath = sys.argv[2]
            applistpath = sys.argv[3]
            thval = sys.argv[4]
            resume = sys.argv[5]
            savetodisk = sys.argv[6]
            ta_class = TextAugmentationClass(outputpath, thval, applistpath)
            ta_class.AugmentText(inputpath, outputpath, resume, savetodisk)
        except BaseException as e:
            print("Text augmentation has failed:" + str(e))
    
        