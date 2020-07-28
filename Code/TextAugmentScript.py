# import tensorflow as tf
import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action
import gensim.models.keyedvectors as word2vec

import numpy as np

def substitute_contextualized_embeddings(df):
    aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="substitute")
    
    for i, j in df.iterrows(): 
        if j.Text != "":
            print(i)
            print(j.Text) 

            augmented_text = aug.augment(j.Text)
            print(augmented_text)
            
            # sentence similarity
            sim = sentence_similarity(augmented_text, j.Text)
            print(sim)
            if (sim > 0.8 and sim <= 1.0):
                j.Text = augmented_text
            else:
                df = df.drop(i)
            print()
    print(df)
    return

def insert_contextualized_embeddings(df):
    aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")
    for i, j in df.iterrows(): 
        if j.Text != "":
            print(i)
            print(j.Text) 

            augmented_text = aug.augment(j.Text)
            print(augmented_text)
            
            # sentence similarity
            sim = sentence_similarity(augmented_text, j.Text)
            print(sim)
            if (sim > 0.8 and sim <= 1.0):
                j.Text = augmented_text
            else:
                df = df.drop(i)
            print()
    print(df)
    return
    
def sentence_similarity(sen_1, sen_2):
    sen_1_words = [w for w in sen_1.split() if w in model.vocab]
    sen_2_words = [w for w in sen_2.split() if w in model.vocab]
    
    if sen_1_words and sen_2_words:
        sim = model.n_similarity(sen_1_words, sen_2_words)
        return sim

if __name__ == "__main__":
    inputpath = r"C:\Users\huangra\Documents\GitHub\UIFtextaugmentation\Data\Test.csv"
    inputData = pd.read_csv(inputpath, encoding="latin1", keep_default_na=False)
    
    model = word2vec.KeyedVectors.load_word2vec_format(r"C:\Users\huangra\Downloads\GoogleNews-vectors-negative300.bin", binary=True)
    #Check column names:
    try:
        col_list = list(inputData.columns)
        if (('Text' not in col_list) or ('ID' not in col_list)):
            raise Exception("Input data should have following columns: Text, ID")
        data = pd.DataFrame(columns = inputData.columns)
    except BaseException as e:
        print("Input data has problem:" + str(e))
    
    df = pd.DataFrame(inputData) 
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print(df)

    # text augmentation techniques:
    substitute_contextualized_embeddings(df)
    insert_contextualized_embeddings(df)
    # write to output file
    df.to_csv(r'C:\Users\huangra\Documents\GitHub\UIFtextaugmentation\Output\Output.csv', index=False, mode='a')



    
    