import pandas as pd
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

import gensim
import gensim.models.keyedvectors as word2vec

import numpy as np

def contextual_embeddings(df,model, context_action):
    aug = naw.ContextualWordEmbsAug(
    model_path=model, action=context_action)

    print("STARTING CONTEXTUAL EMBEDDINGS: ", model, " ",context_action)
    text_augmentation(aug,df)

def embeddings(df, model, context_action):
    aug = naw.WordEmbsAug(
    model_type=model, model_path=r"PATH TO WORD2VEC MODEL",
    action=context_action)

    print("STARTING EMBEDDINGS: ", context_action)
    text_augmentation(aug,df)


def sentence_similarity(sen_1, sen_2):
    sen_1_words = [w for w in sen_1.split() if w in model.vocab]
    sen_2_words = [w for w in sen_2.split() if w in model.vocab]
    
    if sen_1_words and sen_2_words:
        sim = model.n_similarity(sen_1_words, sen_2_words)
        return sim

def text_augmentation(aug,df):
    cols = list(df.columns.values)
    for i, j in df.iterrows(): 
        # remove this if statement to generate data for all categories
        if (j[cols[3]] is 0 and j[cols[2]] is 0): #exclude category 0 and 1
            print(i)
            print(j.Text) 
            augmented_text = aug.augment(j.Text)
            print(augmented_text)
            
            # sentence similarity
            sim = sentence_similarity(augmented_text, j.Text)
            print(sim)
            if sim:
                if (sim > 0.8 and sim < 1.0):
                    df = df.replace(to_replace=j.Text, 
                        value =augmented_text)
                else:
                    df = df.drop(i)
            else:
                    df = df.drop(i)
            print()
        else:
            df = df.drop(i)

    # write to output file
    df.to_csv(r"PATH TO OUTPUT DATA", index=False, header=False, mode="a")
    return

if __name__ == "__main__":
    inputpath = r"PATH TO INPUT DATA"
    inputData = pd.read_csv(inputpath, encoding="latin1", keep_default_na=False)
    
    model = word2vec.KeyedVectors.load_word2vec_format(r"PATH TO WORD2VEC MODEL", binary=True)
    #Check column names:
    try:
        col_list = list(inputData.columns)
        if (("Text" not in col_list) or ("ID" not in col_list)):
            raise Exception("Input data should have following columns: Text, ID")
        data = pd.DataFrame(columns = inputData.columns)
    except BaseException as e:
        print("Input data has problem:" + str(e))
    
    df = pd.DataFrame(inputData) 
    # delete any empty rows
    df.replace("", np.nan, inplace=True)
    df.dropna(inplace=True)


    # text augmentation techniques:

    for i in range(0,20):
        contextual_embeddings(df,'bert-base-uncased',"substitute")
        contextual_embeddings(df,"bert-base-uncased","insert")
        contextual_embeddings(df,"distilbert-base-uncased","substitute")
        contextual_embeddings(df,"roberta-base","substitute")
        embeddings(df, "word2vec", "substitute")
        embeddings(df, "word2vec","insert")
    



    
    