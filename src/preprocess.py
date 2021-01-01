import pandas as pd
import os
import sys
import json

def class Preprocess:
    
    def __init__(self,tokenizer,entity_labels,label_type = 'BOI'):
        self.tokenizer = tokenizer
        self.entity_lables = entity_labels
        self.label_type = label_type        
    
    def read_files(self,path_text,path_tsv):
        texts = []
        text_labels = []
        exceptions = []
        for file in all_files:
            try:
                with open(os.path.join(path_text,file)) as f:
                    text = f.read()
                tsv_data = pd.read_csv(os.path.join(path_tsv,file.split('.')[0]+".tsv"),sep="\t")[['annotType','startOffset','endOffset','text','annotId','other']].sort_values(by='startOffset',ascending=False)
                texts.append(text)
                text_labels.append(tsv_data.values.tolist())
            except Exception as e:
                exceptions.append(e)
        
        return texts,text_labels
    
        
        
        

