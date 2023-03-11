import numpy as np
import pandas as pd

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


############ MyDataset ###############
class MyDataset(Dataset):
    def __init__(self, 
                 df, 
                 tokenizer, 
                 max_length, 
                 mode = "train"):
        
        self.df = df
        self.max_length=  max_length
        self.tokenizer = tokenizer
        self.mode = mode

        # x
        self.text = self.df.new_sentence.values

        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        sentence = self.text[idx]

        inputs = self.tokenizer.encode_plus(sentence, 
                                            add_special_tokens=True,
                                            # padding='max_length', 
                                            max_length = self.max_length, 
                                            truncation=True)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        if self.mode == "train":
            
            # y1: TYPE - 유형
            y_type = self.df['type'].values[idx]

            # y2:  PN - 극성
            y_pn = self.df['pn'].values[idx]

            # y3: time - 시제
            y_time = self.df['time'].values[idx]

            # y4: sure - 확실성
            y_sure = self.df['sure'].values[idx]


            return {'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'target_type': y_type,
                    'target_pn': y_pn,
                    'target_time': y_time,
                    'target_sure': y_sure}

        else:
            return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],}
          
          
          
####### prepare_loaders #########          
def prepare_loader(train, 
                   fold, 
                   tokenizer, 
                   max_length, 
                   bs,
                   collate_fn
                   ):
    
    train_df = train[train.kfold != fold].reset_index(drop=True)
    valid_df = train[train.kfold == fold].reset_index(drop=True)

    ## train, valid -> Dataset
    train_ds = MyDataset(train_df, 
                            tokenizer = tokenizer ,
                            max_length = max_length,
                            mode = "train")

    valid_ds = MyDataset(valid_df, 
                            tokenizer = tokenizer ,
                            max_length = max_length,
                            mode = "train")
    
    # Dataset -> DataLoader
    train_loader = DataLoader(train_ds,
                              batch_size = bs, 
                              collate_fn=collate_fn, 
                              num_workers = 2,
                              shuffle = True, 
                              pin_memory = True, 
                              drop_last= True)

    valid_loader = DataLoader(valid_ds,
                              batch_size = bs,
                              collate_fn=collate_fn,
                              num_workers = 2,
                              shuffle = False, 
                              pin_memory = True,)
    print("DataLoader Completed")
    return train_loader, valid_loader
  
  
  
  
  
