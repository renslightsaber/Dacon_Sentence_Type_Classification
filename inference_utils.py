import os
import gc
import copy
import time

import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from dataloader import *
from model import *

######## test_loader ##############
def make_testloader(test, 
                    tokenizer, 
                    max_length,
                    bs, 
                    collate_fn):

    test_ds = MyDataset(test, 
                        tokenizer = tokenizer,
                        max_length = max_length,
                        mode = "test")

    test_loader = DataLoader(test_ds,
                            batch_size = bs,
                            # num_workers = 2,
                            # pin_memory = True, 
                            collate_fn = collate_fn,
                            shuffle = False, 
                            drop_last= False)
    
    print("TestLoader Completed")
    return test_loader
  

  
  
  
########### Test Function #################
@torch.no_grad()
def test_func(model, dataloader, device):

    type_preds, pn_preds, time_preds, sure_preds = [], [], [], []

    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total = len(dataloader))
        for step, data in bar:
            ids = data['input_ids'].to(device, dtype = torch.long)
            masks = data['attention_mask'].to(device, dtype = torch.long)

            # y_preds
            y_preds = model(ids, masks) 

            type_preds.append(y_preds['type'].detach().cpu().numpy())
            pn_preds.append(y_preds['pn'].detach().cpu().numpy())
            time_preds.append(y_preds['time'].detach().cpu().numpy())
            sure_preds.append(y_preds['sure'].detach().cpu().numpy())

    predictions = dict()

    type_predict = np.concatenate(type_preds, axis= 0)
    pn_predict = np.concatenate(pn_preds, axis= 0)
    time_predict = np.concatenate(time_preds, axis= 0)
    sure_predict = np.concatenate(sure_preds, axis= 0)

    predictions['type'] = type_predict
    predictions['pn'] = pn_predict
    predictions['time'] = time_predict
    predictions['sure'] = sure_predict

    gc.collect()
    
    return predictions
  
  
  
################## Trained Model Save-Path List ########################  
## Better F1 Score Model paths 
def trained_model_paths(n_folds, model_save, model_type):
    print("n_folds: ",n_folds )

    model_paths_f1 = []
    
    # Define Model because of KFold
    if model_type == 1:
        model_type_s = "ModelV1"

    elif model_type == 2:
        model_type_s = "ModelV2"

    elif model_type == 3:
        model_type_s = "ModelV3"

    else:
        model_type_s = "ModelV4"

    for num in range(0, n_folds):
        model_paths_f1.append(model_save + model_type_s + "_" +f"Loss-Fold-{num}_f1.bin")

    print(len(model_paths_f1))
    print(model_paths_f1)
    return model_paths_f1
  
  
#################### inference model define #################################
# Define Model because of KFold
def inference_model_define(model_type, 
                           model_name, 
                           device):
    
    if model_type == 1:
        model = ModelV1(model_name).to(device)
        print("ModelV1")

    elif model_type== 2:
        model = ModelV2(model_name).to(device)
        print("ModelV2")

    elif model_type == 3:
        model = ModelV3(model_name).to(device)
        print("ModelV3")

    else:
        model = ModelV4(model_name).to(device)
        print("ModelV4")

    return model
  
  
################# Inference ###################
def inference(model_paths, 
              model_type, 
              model_name, 
              dataloader, 
              device):

    final_type_preds, final_pn_preds, final_time_preds, final_sure_preds = [], [], [], []
    
    for i, path in enumerate(model_paths):
        model = inference_model_define(model_type, model_name, device)
        model.load_state_dict(torch.load(path))
        
        print(f"Getting predictions for model {i+1}")
        preds = test_func(model, dataloader, device)

        # 줍줍
        # final_preds.append(preds)
        final_type_preds.append(preds['type'])
        final_pn_preds.append(preds['pn'])
        final_time_preds.append(preds['time'])
        final_sure_preds.append(preds['sure'])

    
    # 그리고 평균을 내줍니다.
    # final_preds = np.array(final_preds)
    # final_preds = np.mean(final_preds, axis=0)

    # TYPE: 유형
    final_type_preds = np.array(final_type_preds)
    final_type_preds = np.mean(final_type_preds, axis = 0)

    # PN: 극성
    final_pn_preds = np.array(final_pn_preds)
    final_pn_preds = np.mean(final_pn_preds, axis = 0)

    # TIME: 시제
    final_time_preds = np.array(final_time_preds)
    final_time_preds = np.mean(final_time_preds, axis = 0)

    # SURE: 확실성
    final_sure_preds = np.array(final_sure_preds)
    final_sure_preds = np.mean(final_sure_preds, axis = 0)

    final_preds = dict()
    final_preds['type'] = final_type_preds
    final_preds['pn'] = final_pn_preds
    final_preds['time'] = final_time_preds
    final_preds['sure'] = final_sure_preds
    
    gc.collect()

    return final_preds
  
  
################# COLUMN WISE Predict #####################  
def column_wise_predict(f1_preds, 
                        ss, 
                        column, 
                        inverse_encode, 
                        threshold = .5):
  
    # ss = submission file
    print("column_name = 'time: ", column)
 
    if column == 'sure':
        class_index = f1_preds[column] > threshold
    else:
        class_index = np.argmax(f1_preds[column], axis = 1)
        print("Shape of preds: ", class_index.shape)

    print()
    print(ss.shape)
    ss[column] = class_index
    print(ss[column].value_counts())
    print()
    ss[column] = ss[column].apply(lambda x: inverse_encode[column][x])
    print(ss.shape)
    print(ss[column].value_counts())
    print()
  
  
  
  
