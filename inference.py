import re
import os
import gc
import random
import string

import argparse
import ast

import copy
from copy import deepcopy

import torchmetrics
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAccuracy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

## Pytorch Import
import torch 
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

## Transforemr Import
from transformers import AutoTokenizer, AutoModel, AdamW, AutoConfig, DataCollatorWithPadding

# Utils
from tqdm.auto import tqdm, trange

import time
from time import sleep

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


from dataloader import *
from new_trainer import *
from model import *
from utils import *
from inference_utils import *


def define():
    p = argparse.ArgumentParser()

    p.add_argument('--base_path', type = str, default = "./data/", help="Data Folder Path")
    
    ## 다른 경로로 하는 것이 맞
    p.add_argument('--model_save', type = str, default = "./data/", help="Data Folder Path")
    p.add_argument('--sub_path', type = str, default = "./data/", help="Data Folder Path")
    
    p.add_argument('--clean_text', type = bool, default = False, help="Mecab Tokenized -> " ".join")
    p.add_argument('--test_and_ss', type = bool, default = True, help="test.csv, sample_submission.csv")
    
    p.add_argument('--model', type = str, default = "monologg/kobigbird-bert-base", help="HuggingFace Pretrained Model")
    p.add_argument('--model_type', type = int, default = 1, help="ModelV")
    
    p.add_argument('--n_folds', type = int, default = 5, help="Folds")
    
    p.add_argument('--seed', type = int, default = 2022, help="Seed")
    p.add_argument('--train_bs', type = int, default = 16, help="Batch Size")
    
    p.add_argument('--max_length', type = int, default = 128, help="Max Length")
    p.add_argument('--device', type = str, default = "cuda", help="CUDA or MPS or CPU?")

    config = p.parse_args()
    return config
  
def main(config):
    
    ## Set Seed
    set_seed(config.seed)
    print("Seed")
    
    ## Data
    train, test, ss = dacon_competition_data(base_path = config.base_path, 
                                             clean_text = config.clean_text, 
                                             test_and_ss = config.test_and_ss)
    
    
    ## Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    print("Tokenizer Downloaded")

    # Device
    if config.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
    elif config.device == "cuda":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
    else:
        device = torch.device("cpu")
        
        
    
    
    ## DataLoader
    test_loader = make_testloader(test, 
                                  tokenizer, 
                                  config.max_length, 
                                  config.train_bs, 
                                  collate_fn= DataCollatorWithPadding(tokenizer=tokenizer))
    
    ## Saved Models Path
    model_paths_f1 = trained_model_paths(n_folds = config.n_folds, 
                                         model_save = config.model_save, 
                                         model_type = config.model_type)
    
    ## Inference
    f1_preds = inference(model_paths_f1, 
                         test_loader, 
                         device)
    print("Inference Completed")
    
    
    ## Target Decoder
    target4_inverse = {True: '확실', False: '불확실'}
    inverse_encode = {'type': target1_inverse, 
                      'pn': target2_inverse, 
                      'time': target3_inverse, 
                      'sure': target4_inverse, }
    print("Targer Decoders")
    
    
    # 유형 라벨 디코딩
    print("유형")
    column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'type', )
    print(ss.head(3))
    
    # 극성 라벨 디코딩
    print("극성")
    column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'pn',)
    print(ss.head(3))
    
    # 시제 라벨 디코딩
    print("시제")
    column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'time')
    print(ss.head(3))
    
    # 확실성 라벨 디코딩
    print("확실성")
    column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'sure')
    print(ss.head(3))
    
    # SUM
    ss.drop(['label'], axis = 1, inplace = True)
    print("Drop 'label' Column and Before SUM")
    print(ss.shape)
    print(ss.head(3))
    
    ss.loc[:, 'label'] = ss.loc[:, 'type'] + '-' + ss.loc[:, 'pn'] + '-' + ss.loc[:, 'time'] + '-' + ss.loc[:,'sure']
    print("After SUM")
    print(ss.shape)
    print(ss.head(3))
    
    # Drop Columns : 'type', 'pn', 'time', 'sure'
    ss.drop(['type', 'pn', 'time', 'sure'], axis = 1, inplace = True)
    print("Drop Columns")
    print(ss.shape)
    ss.head()
    
    ## Submission csv file name and save
    sub_file_name = config.sub_path + "_".join(config.model.split("/"))+ "_ModelV" + str(config.model_type) + "_N_folds" + str(config.n_folds) + "_N_epochs" + str(config.n_epochs) + "_LR" + str(config.learning_rate) + ".csv"
    print(sub_file_name)
    
    print("Save Submission.csv")
          
         
    ### K Fold: MultilabelStratifiedKFold
    skf = MultilabelStratifiedKFold(n_splits = config.n_folds, 
                                    shuffle = True, 
                                    random_state = config.seed)

    for fold, (_, val_index) in enumerate(skf.split(X=train, y=train[target_cols])):
        train.loc[val_index, 'kfold'] = int(fold)

    train['kfold'] = train['kfold'].astype('int')
    ## Drop Unnecessary Cols
    train.drop(['유형', '극성', '시제', '확실성', 'label'], axis = 1, inplace = True)
    print(train.shape)
    print(train.head(3))
    
    ## Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    print("Tokenizer Downloaded")

    # Device
    if config.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
    elif config.device == "cuda":
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
    else:
        device = torch.device("cpu")

    print("Device", device)
    
    ### folds_run
    best_scores = []
    
    for fold in trange(0, config.n_folds, desc='Fold Loop'):
    # for fold in trange(0, config['n_folds']):

        print(f"{y_}==== Fold: {fold} ====={sr_}")

        # DataLoaders 
        train_loader, valid_loader = prepare_loader(train, 
                                                    fold, 
                                                    tokenizer, 
                                                    config.max_length, 
                                                    config.train_bs, 
                                                    DataCollatorWithPadding(tokenizer=tokenizer))

        ## Define Model because of KFold
        if config.model_type == 1:
            model = ModelV1(config.model).to(device)
            # print("ModelV1")

        elif config.model_type == 2:
            model = ModelV2(config.model).to(device)
            # print("ModelV2")

        elif config.model_type == 3:
            model = ModelV3(config.model).to(device)
            # print("ModelV3")

        else:
            model = ModelV4(config.model).to(device)
            # print("ModelV4")

        # Loss Function
        loss_fn = {'type': nn.NLLLoss().to(device),
                    'pn' : nn.NLLLoss().to(device),
                    'time': nn.NLLLoss().to(device),
                    'sure': nn.BCELoss().to(device)}
        print("Loss Function Defined")

        # Define Opimizer and Scheduler
        optimizer = AdamW(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
        print("Optimizer Defined")

        # scheduler = fetch_scheduler(optimizer)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)
        print("Scheduler Defined")
        
        print("자세히 알고 싶으면 코드를 봅시다.")
        ## Start Training
        model, best_score = run_train(model, 
                                      config.model_type, 
                                      config.model_save, 
                                      train_loader, 
                                      valid_loader, 
                                      loss_fn, 
                                      optimizer, 
                                      device, 
                                      n_classes, 
                                      fold, 
                                      scheduler, 
                                      config.grad_clipping, 
                                      config.n_epochs)

        ## Best F1_Score per Fold 줍줍
        if type(best_score) == torch.Tensor:
            best_scores.append( best_score.detach().cpu().item() )
        else:
            best_scores.append(best_score)
        
        ## For Memory
        del model, train_loader, valid_loader

        torch.cuda.empty_cache()
        _ = gc.collect()

    print(best_scores)
    print("Train Completed")
    
    

if __name__ == '__main__':
    config = define()
    main(config)
    
    
    
    
