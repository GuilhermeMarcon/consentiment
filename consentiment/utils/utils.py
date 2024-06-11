
import pandas as pd
import numpy as np
from ast import literal_eval
import torch
from transformers import BertTokenizer, BertModel

def read_df(filename:str, clean_empty:bool=True, verbose:bool=True):
    if(verbose): print("Reading", filename)
    df = pd.read_excel(filename)
    if(clean_empty and verbose): print("Original size", len(df))
    for col in df.columns[1:]:
        df[col] = df[col].apply(literal_eval)
        if(clean_empty): df = df[df[col].apply(len) != 0]
    if(verbose): print("Final size", len(df))
    return df

from tqdm import tqdm
tqdm.pandas()

def create_embeddings(df:pd.DataFrame, text:str='review', embedding_model:str="bert-base-multilingual-cased", verbose:bool=True):
    # Load pre-trained model/tokenizer
    tokenizer = BertTokenizer.from_pretrained(embedding_model)
    model = BertModel.from_pretrained(embedding_model)

    def text_to_embedding(text):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens_tensor = torch.tensor([tokens])
        with torch.no_grad():
            outputs = model(tokens_tensor)
            embedding = outputs.pooler_output.squeeze(0).cpu().detach().numpy()
        return list(embedding)

    def entities_to_embeddings(entities):
        return [text_to_embedding(x[0]) for x in entities]

    if(verbose): print("Embedding sentences...")
    original_cols = list(df.columns)
    df[text+'_emb'] = df[text].progress_apply(text_to_embedding)
    for col in original_cols[1:]:
        if col == 'annotation': continue
        if col.endswith('_emb'): continue
        if(verbose): print('Embedding entities "'+str(col)+'" ...')
        df[col+'_ents_emb'] = df[col].progress_apply(entities_to_embeddings)
