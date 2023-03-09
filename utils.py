import numpy as np
import pandas as pd

import torch
import torch.nn as nn
# Scheduler
import torch.optim as optim

import matplotlib.pyplot as plt

from konlpy.tag import Mecab



############# seed #################
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
############# make_class_weights ##############   
def make_class_weights(df, column = 'type', device = config['device'] ):
    # class imbalance 해결
    # Target의 데이터 수를 class number로 오름차순 정렬
    class_counts = df[column].value_counts()
    class_weights = 1./class_counts
    class_weights = class_weights/class_weights.min()
    class_weights = class_weights.to_dict()
    class_weights = {k: v for k, v in sorted(class_weights.items(), key=lambda item: item[0])}
    class_weights = list(class_weights.values())
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(class_weights)
    return class_weights

  
######### Data #############
def dacon_competition_data( base_path = './data/', clean_text = False, test_and_ss = False):
    
    if clean_text:
        train = pd.read_csv(base_path + 'train.csv')
        mecab = Mecab() 
        train['new_sentence'] = train["문장"].apply(lambda x: " ".join(mecab.morphs(x)))
        print(train.head())
    else:
        train = pd.read_csv(base_path + 'train.csv')
        # Column Rename
        train.rename(columns={'문장':'new_sentence'}, inplace = True)
        print(train.head())

    print("Data Shape: ", train.shape)

    # print("Nunique of Target Categories: ", train.iloc[:, -1].nunique())
    # print("Values of Target Categories: ", train.iloc[:, -1].unique())

    if test_and_ss:
        test = pd.read_csv(base_path + 'test.csv')
        # Column Rename
        test.rename(columns={'문장':'new_sentence'}, inplace = True)
        ss = pd.read_csv(base_path + 'sample_submission.csv')
        return train, test, ss
    else:
        return train  
      
########## make_ plot ###########      
def make_plot(result, stage = "Loss"):

    plot_from = 0

    if stage == "Loss":
        trains = 'Train Loss'
        valids = 'Valid Loss'

    elif stage == "Acc":
        trains = "Train Acc"
        valids = "Valid Acc"

    elif stage == "F1":
        trains = "Train F1"
        valids = "Valid F1"

    plt.figure(figsize=(10, 6))
    
    plt.title(f"Train/Valid {stage} History", fontsize = 20)

    ## Modified for converting Type
    if type(result[trains][0]) == torch.Tensor:
        result[trains] = [num.detach().cpu().item() for num in result[trains]]
        result[valids] = [num.detach().cpu().item() for num in result[valids]]

    plt.plot(
        range(0, len(result[trains][plot_from:])), 
        result[trains][plot_from:], 
        label = trains
        )

    plt.plot(
        range(0, len(result[valids][plot_from:])), 
        result[valids][plot_from:], 
        label = valids
        )

    plt.legend()
    if stage == "Loss":
        plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    
      
########### Scheduler #################
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor    
