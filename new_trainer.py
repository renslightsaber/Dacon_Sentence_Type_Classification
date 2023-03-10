import re
import os
import gc
import random

import copy
from copy import deepcopy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

import torchmetrics
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAccuracy

# About tqdm: https://github.com/tqdm/tqdm/#ipython-jupyter-integration
from tqdm.auto import tqdm, trange

import time
from time import sleep

########### Train One Epoch() #####################
def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, n_classes, scheduler = None, grad_clipping = False):


    ################ torchmetrics: initialize metric #########################
    
    metric_acc_type = torchmetrics.Accuracy(task='multiclass', average = 'weighted', num_classes=n_classes['type']).to(device)
    metric_f1_type = torchmetrics.F1Score(task="multiclass", average = 'weighted', num_classes=n_classes['type']).to(device)

    metric_acc_pn = torchmetrics.Accuracy(task='multiclass', average = 'weighted', num_classes=n_classes['pn']).to(device)
    metric_f1_pn = torchmetrics.F1Score(task="multiclass", average = 'weighted',num_classes=n_classes['pn']).to(device)

    metric_acc_time = torchmetrics.Accuracy(task='multiclass', average = 'weighted', num_classes=n_classes['time']).to(device)
    metric_f1_time = torchmetrics.F1Score(task="multiclass", average = 'weighted',num_classes=n_classes['time']).to(device)

    metric_bi_acc = BinaryAccuracy().to(device)
    metric_bi_f1 = BinaryF1Score().to(device)
    
    ############################################################################

    train_loss = 0
    dataset_size = 0

    bar = tqdm(enumerate(dataloader), total = len(dataloader), desc='Train Loop')
    # bar = tqdm_notebook(enumerate(dataloader), total = len(dataloader), desc='Train Loop', leave=False)

    model.train()
    for step, data in bar:

        ids = data['input_ids'].to(device, dtype = torch.long)
        masks = data['attention_mask'].to(device, dtype = torch.long)
        # targets
        t_type = data['target_type'].to(device, dtype = torch.long)
        t_pn = data['target_pn'].to(device, dtype = torch.long)
        t_time = data['target_time'].to(device, dtype = torch.long)
        t_sure = data['target_sure'].to(device, dtype = torch.float)
 
        # y_preds
        y_preds = model(ids, masks) 

        # Loss
        loss1 = loss_fn['type'](y_preds['type'], t_type )
        loss2 = loss_fn['pn'](y_preds['pn'], t_pn )
        loss3 = loss_fn['time'](y_preds['time'], t_time )
        # binary acc, f1_score
        y_preds['sure'] = y_preds['sure'].view(-1)
        loss4 = loss_fn['sure'](y_preds['sure'], t_sure)


        # loss sum
        loss = (loss1 + loss2 + loss3 + loss4) / 4

        optimizer.zero_grad()
        loss.backward()
        

        # Gradient-Clipping | source: https://velog.io/@seven7724/Transformer-계열의-훈련-Tricks
        max_norm = 5
        if grad_clipping:
            #print("Gradient Clipping Turned On | max_norm: ", max_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        batch_size = ids.size(0)
        dataset_size += batch_size
        train_loss += float(loss.item() * batch_size) 
        train_epoch_loss = train_loss / dataset_size 
        
        # Type - ACC, F1
        acc_type = metric_acc_type(y_preds['type'], t_type )
        acc_type = acc_type.detach().cpu().item()
        f1_type = metric_f1_type(y_preds['type'], t_type )
        f1_type = f1_type.detach().cpu().item()

        # PN - ACC, F1
        acc_pn = metric_acc_pn(y_preds['pn'], t_pn )
        acc_pn = acc_pn.detach().cpu().item()
        f1_pn = metric_f1_pn(y_preds['pn'], t_pn )
        f1_pn = f1_pn.detach().cpu().item()

        # TIME - ACC, F1
        acc_time = metric_acc_time(y_preds['time'], t_time )
        acc_time = acc_time.detach().cpu().item()
        f1_time = metric_f1_time(y_preds['time'], t_time )
        f1_time = f1_time.detach().cpu().item()

        # SURE - ACC, F1
        acc_sure = metric_bi_acc(y_preds['sure'], t_sure )
        acc_sure = acc_sure.detach().cpu().item()
        f1_sure = metric_bi_f1(y_preds['sure'], t_sure )
        f1_sure = f1_sure.detach().cpu().item()
        
        bar.update()
        bar.set_postfix(Epoch = epoch, 
                        Train_loss = train_epoch_loss,
                        LR = optimizer.param_groups[0]['lr'])
        
    # Type - ACC, F1
    acc_type = metric_acc_type.compute()
    f1_type = metric_f1_type.compute()

    # PN - ACC, F1
    acc_pn = metric_acc_pn.compute()
    f1_pn = metric_f1_pn.compute()

    # TIME - ACC, F1
    acc_time = metric_acc_time.compute()
    f1_time = metric_f1_time.compute()

    # SURE - ACC, F1
    acc_sure = metric_bi_acc.compute()
    f1_sure = metric_bi_f1.compute()


    print(f"Train | Type' | PosNg | Time' | Sure' |")
    print(f"ACCUR | {acc_type:.3f} | {acc_pn:.3f} | {acc_time:.3f} | {acc_sure:.3f} |")
    print(f"F1_SC | {f1_type:.3f} | {f1_pn:.3f} | {f1_time:.3f} | {f1_sure:.3f} |")

    acc_metric = {'type': acc_type, 'pn' : acc_pn,'time': acc_time,'sure': acc_sure}
    f1_metric = {'type': f1_type, 'pn' : f1_pn,'time': f1_time,'sure': f1_sure}

    del acc_type, acc_pn, acc_time, acc_sure, f1_type, f1_pn, f1_time, f1_sure

    # Reseting internal state such that metric ready for new data
    metric_acc_type.reset()
    metric_acc_pn.reset()
    metric_acc_time.reset()
    metric_bi_acc.reset()

    metric_f1_type.reset()
    metric_f1_pn.reset()
    metric_f1_time.reset()
    metric_bi_f1.reset()

    torch.cuda.empty_cache()
    _ = gc.collect()

    return train_epoch_loss, acc_metric, f1_metric
  
  

  
  
############ Valid One Epoch() ######################
# Valid One Epoch
@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, n_classes):


    ################ torchmetrics: initialize metric #########################
    
    metric_acc_type = torchmetrics.Accuracy(task='multiclass', average = 'weighted', num_classes=n_classes['type']).to(device)
    metric_f1_type = torchmetrics.F1Score(task="multiclass", average = 'weighted', num_classes=n_classes['type']).to(device)

    metric_acc_pn = torchmetrics.Accuracy(task='multiclass', average = 'weighted', num_classes=n_classes['pn']).to(device)
    metric_f1_pn = torchmetrics.F1Score(task="multiclass", average = 'weighted',num_classes=n_classes['pn']).to(device)

    metric_acc_time = torchmetrics.Accuracy(task='multiclass', average = 'weighted', num_classes=n_classes['time']).to(device)
    metric_f1_time = torchmetrics.F1Score(task="multiclass", average = 'weighted',num_classes=n_classes['time']).to(device)

    metric_bi_acc = BinaryAccuracy().to(device)
    metric_bi_f1 = BinaryF1Score().to(device)
    
    ############################################################################
    
    valid_loss = 0
    dataset_size = 0
    
    #tqdm의 경우, for문에서 iterate할 때 실시간으로 보여주는 라이브러리입니다. 보시면 압니다. 
    bar = tqdm(enumerate(dataloader), total = len(dataloader), desc='Valid Loop')
    # bar = tqdm_notebook(enumerate(dataloader), total = len(dataloader), desc='Valid Loop', leave=False)

    model.eval()
    with torch.no_grad():
        for step, data in bar:

            ids = data['input_ids'].to(device, dtype = torch.long)
            masks = data['attention_mask'].to(device, dtype = torch.long)

            # targets
            t_type = data['target_type'].to(device, dtype = torch.long)
            t_pn = data['target_pn'].to(device, dtype = torch.long)
            t_time = data['target_time'].to(device, dtype = torch.long)
            t_sure = data['target_sure'].to(device, dtype = torch.float)
    
            # y_preds
            y_preds = model(ids, masks) 

            # Loss
            loss1 = loss_fn['type'](y_preds['type'], t_type )
            loss2 = loss_fn['pn'](y_preds['pn'], t_pn )
            loss3 = loss_fn['time'](y_preds['time'], t_time )

            # binary acc, f1_score
            y_preds['sure'] = y_preds['sure'].view(-1)
            loss4 = loss_fn['sure'](y_preds['sure'], t_sure)


            # loss sum
            loss = (loss1 + loss2 + loss3 + loss4) / 4

            # 실시간 Loss
            batch_size = ids.size(0)
            dataset_size += batch_size
            valid_loss += float(loss.item() * batch_size)
            valid_epoch_loss = valid_loss / dataset_size

            # Type - ACC, F1
            acc_type = metric_acc_type(y_preds['type'], t_type )
            acc_type = acc_type.detach().cpu().item()
            f1_type = metric_f1_type(y_preds['type'], t_type )
            f1_type = f1_type.detach().cpu().item()

            # PN - ACC, F1
            acc_pn = metric_acc_pn(y_preds['pn'], t_pn )
            acc_pn = acc_pn.detach().cpu().item()
            f1_pn = metric_f1_pn(y_preds['pn'], t_pn )
            f1_pn = f1_pn.detach().cpu().item()

            # TIME - ACC, F1
            acc_time = metric_acc_time(y_preds['time'], t_time )
            acc_time = acc_time.detach().cpu().item()
            f1_time = metric_f1_time(y_preds['time'], t_time )
            f1_time = f1_time.detach().cpu().item()

            # SURE - ACC, F1
            acc_sure = metric_bi_acc(y_preds['sure'], t_sure )
            acc_sure = acc_sure.detach().cpu().item()
            f1_sure = metric_bi_f1(y_preds['sure'], t_sure )
            f1_sure = f1_sure.detach().cpu().item()
            
            bar.update()
            bar.set_postfix(Epoch = epoch, 
                            Valid_loss = valid_epoch_loss,
                            LR = optimizer.param_groups[0]['lr'],
                            )

    # Type - ACC, F1
    acc_type = metric_acc_type.compute()
    f1_type = metric_f1_type.compute()

    # PN - ACC, F1
    acc_pn = metric_acc_pn.compute()
    f1_pn = metric_f1_pn.compute()

    # TIME - ACC, F1
    acc_time = metric_acc_time.compute()
    f1_time = metric_f1_time.compute()

    # SURE - ACC, F1
    acc_sure = metric_bi_acc.compute()
    f1_sure = metric_bi_f1.compute()


    print(f"Valid | Type' | PosNg | Time' | Sure' |")
    print(f"ACCUR | {acc_type:.3f} | {acc_pn:.3f} | {acc_time:.3f} | {acc_sure:.3f} |")
    print(f"F1_SC | {f1_type:.3f} | {f1_pn:.3f} | {f1_time:.3f} | {f1_sure:.3f} |")
    print()

    acc_metric = {'type': acc_type, 'pn' : acc_pn,'time': acc_time,'sure': acc_sure}
    f1_metric = {'type': f1_type, 'pn' : f1_pn,'time': f1_time,'sure': f1_sure}

    del acc_type, acc_pn, acc_time, acc_sure, f1_type, f1_pn, f1_time, f1_sure

    # Reseting internal state such that metric ready for new data
    metric_acc_type.reset()
    metric_acc_pn.reset()
    metric_acc_time.reset()
    metric_bi_acc.reset()

    metric_f1_type.reset()
    metric_f1_pn.reset()
    metric_f1_time.reset()
    metric_bi_f1.reset()

    torch.cuda.empty_cache()
    _ = gc.collect()

    return valid_epoch_loss, acc_metric, f1_metric

  
  
  
############### Run Train ##########################
# Run Train
def run_train(model, model_type, model_save, train_loader, valid_loader, loss_fn, optimizer, device, n_classes, fold, scheduler = None, grad_clipping = False, n_epochs=5):
    #, print_iter=1, early_stop=1):
    
    if torch.cuda.is_available():
        print("INFO: GPU - {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict()) 
    # inference with models which is saved at best_score, or lowest_loss updated! 
    # Don't Need to save bst_model_wts like above

    lowest_epoch = np.inf
    lowest_loss = np.inf

    train_hs, valid_hs, train_f1s, valid_f1s = [], [], [], []
    
    best_score = 0
    best_score_epoch = np.inf
    best_model = None


    # Define Model because of KFold
    if model_type == 1:
        model_type_s = "ModelV1"

    elif model_type == 2:
        model_type_s = "ModelV2"

    elif model_type == 3:
        model_type_s = "ModelV3"

    else:
        model_type_s = "ModelV4"
    
    for epoch in range(1, n_epochs + 1):
        gc.collect()

        train_epoch_loss, train_acc_metric, train_f1_metric = train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, n_classes, scheduler, grad_clipping)
        valid_epoch_loss, valid_acc_metric, valid_f1_metric = valid_one_epoch(model, valid_loader, loss_fn, optimizer, device, epoch, n_classes)

        # Mean Weighted F1_score
        train_f1 = sum(train_f1_metric.values()) / len(train_f1_metric.values())
        valid_f1 = sum(valid_f1_metric.values()) / len(valid_f1_metric.values())
        
        ## 줍줍
        train_hs.append(train_epoch_loss)
        valid_hs.append(valid_epoch_loss)

        train_f1s.append(train_f1)
        valid_f1s.append(valid_f1)

        print()
        print(f"Epoch:{epoch:02d} | TL:{train_epoch_loss:.3e} | VL:{valid_epoch_loss:.3e} | Train's F1: {train_f1:.3f} | Valid's F1: {valid_f1:.3f} | ")
        print()

        if valid_epoch_loss < lowest_loss:
            print(f"{b_}Validation Loss Improved({lowest_loss:.3e}) --> ({valid_epoch_loss:.3e})")
            lowest_loss = valid_epoch_loss
            lowest_epoch = epoch
            # best_model_wts = copy.deepcopy(model.state_dict())
            # PATH = model_save + f"Loss-Fold-{fold}.bin"
            # torch.save(model.state_dict(), PATH)
            # print(f"Better Loss Model Saved{sr_}")

        if best_score < valid_f1:
            print(f"{b_}F1 Improved({best_score:.3f}) --> ({valid_f1:.3f})")
            best_score = valid_f1
            best_score_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH2 = model_save + model_type_s + "_" + f"Loss-Fold-{fold}_f1.bin"
            torch.save(model.state_dict(), PATH2)
            print(f"Better_F1_Model Saved{sr_}")
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss : %.4e at %d th Epoch of %dth Fold" % (lowest_loss, lowest_epoch, fold))
    print("Best F1(W): %.4f at %d th Epoch of %dth Fold" % (best_score, best_score_epoch, fold))

    # load best model weights
    model.load_state_dict(best_model_wts)

    result = dict()
    result["Train Loss"] = train_hs
    result["Valid Loss"] = valid_hs

    result["Train F1"] = train_f1s
    result["Valid F1"] = valid_f1s

    # plot
    make_plot(result, stage = "Loss")
    make_plot(result, stage = "F1")
    
    del result, train_hs, valid_hs, train_f1s, valid_f1s 

    torch.cuda.empty_cache()
    _ = gc.collect()

    return model, best_score
  
  
  
  
  
  
  
