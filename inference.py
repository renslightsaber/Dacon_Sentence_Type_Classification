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
    
    ## Target Encoding
    n_classes = dict()
    
    # 유형 라벨 인코딩
    print("유형")
    target1_encode = {v: k for k, v in enumerate(train["유형"].unique())}
    target1_inverse = {v: k for k, v in target1_encode.items()}
    train['type'] = train["유형"].apply(lambda x: target1_encode[x])
    n_classes['type'] = len(target1_encode.keys())
    
    # 극성 라벨 인코딩
    print("극성")
    target2_encode = {v: k for k, v in enumerate(train["극성"].unique())}
    target2_inverse = {v: k for k, v in target2_encode.items()}
    train['pn'] = train["극성"].apply(lambda x: target2_encode[x])
    n_classes['pn'] = len(target2_encode.keys())
    
    # 시제 라벨 인코딩
    print("시제")
    target3_encode = {v: k for k, v in enumerate(train["시제"].unique())}
    target3_inverse = {v: k for k, v in target3_encode.items()}
    train['time'] = train["시제"].apply(lambda x: target3_encode[x])
    n_classes['time'] = len(target3_encode.keys())
    
    # 확실성 라벨 인코딩
    print("확실성")
    train['sure'] = train["확실성"].apply(lambda x: 1 if x == '확실' else 0)
    target4_inverse = {True: '확실', False: '불확실'}
    n_classes['sure'] = 2
    
    
    ## Target Decoder
    target4_inverse = {True: '확실', False: '불확실'}
    inverse_encode = {'type': target1_inverse, 
                      'pn': target2_inverse, 
                      'time': target3_inverse, 
                      'sure': target4_inverse, }
    print("Targer Decoders")
    
    
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
                         config.model_type, 
                         config.model, 
                         test_loader,
                         device)
    print("Inference Completed")
    
    # 유형 라벨 디코딩
    print("유형")
    column_wise_predict(f1_preds= f1_preds, ss = ss, column = 'type', inverse_encode = inverse_encode, threshold = .5)

                        
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
    

if __name__ == '__main__':
    config = define()
    main(config)
    
    
    
    
