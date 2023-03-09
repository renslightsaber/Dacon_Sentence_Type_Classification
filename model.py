import re
import os
import gc
import time
import random

import numpy as np
import pandas as pd

## Pytorch Import
import torch 
import torch.nn as nn

## Transforemr Import
from transformers import AutoModel, AutoConfig


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
      
      
#### Version 01: ModelV1 #####
class ModelV1(nn.Module):
    def __init__(self, model_name):
        super(ModelV1, self).__init__()
        if model_name == 'monologg/kobigbird-bert-base':
            self.model = AutoModel.from_pretrained(model_name, attention_type="original_full")
        else: 
            self.model = AutoModel.from_pretrained(model_name) 
        self.config = AutoConfig.from_pretrained(model_name)
        # self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

        # TYPE: 유형
        self.fc_type = nn.Sequential( nn.Linear(512, 4), nn.LogSoftmax(dim=-1) )

        # PN: 극성
        self.fc_pn = nn.Sequential( nn.Linear(512, 3), nn.LogSoftmax(dim=-1) )

        # TIME: 시제
        self.fc_time = nn.Sequential( nn.Linear(512, 3), nn.LogSoftmax(dim=-1) )

        # SURE: 확실성 - Binary Classification
        self.fc_sure = nn.Sequential( nn.Linear(512, 1), nn.Sigmoid() )
        

    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.relu(self.bn(self.fc(out)))
        
        # TYPE: 유형
        out_type = self.fc_type(out)

        # PN: 극성
        out_pn = self.fc_pn(out)

        # TIME: 시제
        out_time = self.fc_time(out)

        # SURE: 확실성 - Binary Classification
        out_sure = self.fc_sure(out)
        
        outputs = {'type': out_type, 'pn': out_pn, 'time': out_time, 'sure': out_sure}

        return outputs
      
      
      
#### Version 02: ModelV2 #####
class ModelV2(nn.Module):
    def __init__(self, model_name):
        super(ModelV2, self).__init__()
        if model_name == 'monologg/kobigbird-bert-base':
            self.model = AutoModel.from_pretrained(model_name, attention_type="original_full")
        else: 
            self.model = AutoModel.from_pretrained(model_name) 
        self.config = AutoConfig.from_pretrained(model_name)
        # self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

        # TYPE: 유형
        self.fc_type = nn.Sequential( nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 4), nn.LogSoftmax(dim=-1) )

        # PN: 극성
        self.fc_pn = nn.Sequential( nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 3), nn.LogSoftmax(dim=-1) )

        # TIME: 시제
        self.fc_time = nn.Sequential(  nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 3), nn.LogSoftmax(dim=-1) )

        # SURE: 확실성 - Binary Classification
        self.fc_sure = nn.Sequential(  nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid() )
        

    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.relu(self.bn(self.fc(out)))
        
        # TYPE: 유형
        out_type = self.fc_type(out)

        # PN: 극성
        out_pn = self.fc_pn(out)

        # TIME: 시제
        out_time = self.fc_time(out)

        # SURE: 확실성 - Binary Classification
        out_sure = self.fc_sure(out)
        
        outputs = {'type': out_type, 'pn': out_pn, 'time': out_time, 'sure': out_sure}

        return outputs 
      
      
      
#### Version 03: ModelV3 ####
class ModelV3(nn.Module):
    def __init__(self, model_name):
        super(ModelV3, self).__init__()
        if model_name == 'monologg/kobigbird-bert-base':
            self.model = AutoModel.from_pretrained(model_name, attention_type="original_full")
        else: 
            self.model = AutoModel.from_pretrained(model_name) 
        self.config = AutoConfig.from_pretrained(model_name)
        # self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU()

        # TYPE: 유형
        self.fc_type = nn.Sequential( nn.Linear(512, 4), nn.LogSoftmax(dim=-1) )

        # PN: 극성
        self.fc_pn = nn.Sequential( nn.Linear(512, 3), nn.LogSoftmax(dim=-1) )

        # TIME: 시제
        self.fc_time = nn.Sequential( nn.Linear(512, 3), nn.LogSoftmax(dim=-1) )

        # SURE: 확실성 - Binary Classification
        self.fc_sure = nn.Sequential( nn.Linear(512, 1), nn.Sigmoid() )
        

    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.relu(self.bn(self.fc(out)))
        
        # TYPE: 유형
        out_type = self.fc_type(out)

        # PN: 극성
        out_pn = self.fc_pn(out)

        # TIME: 시제
        out_time = self.fc_time(out)

        # SURE: 확실성 - Binary Classification
        out_sure = self.fc_sure(out)
        
        outputs = {'type': out_type, 'pn': out_pn, 'time': out_time, 'sure': out_sure}

        return outputs
      
      
      
#### Version 04: ModelV4 #####
class ModelV4(nn.Module):
    def __init__(self, model_name):
        super(ModelV4, self).__init__()
        if model_name == 'monologg/kobigbird-bert-base':
            self.model = AutoModel.from_pretrained(model_name, attention_type="original_full")
        else: 
            self.model = AutoModel.from_pretrained(model_name) 
        self.config = AutoConfig.from_pretrained(model_name)
        # self.drop = nn.Dropout(p=0.2)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 512)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU()

        # TYPE: 유형
        self.fc_type = nn.Sequential( nn.Linear(512, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Linear(64, 4), nn.LogSoftmax(dim=-1) )

        # PN: 극성
        self.fc_pn = nn.Sequential( nn.Linear(512, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Linear(64, 3), nn.LogSoftmax(dim=-1) )

        # TIME: 시제
        self.fc_time = nn.Sequential(  nn.Linear(512, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Linear(64, 3), nn.LogSoftmax(dim=-1) )

        # SURE: 확실성 - Binary Classification
        self.fc_sure = nn.Sequential(  nn.Linear(512, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Linear(64, 1), nn.Sigmoid() )
        

    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, mask)
        out = self.relu(self.bn(self.fc(out)))
        
        # TYPE: 유형
        out_type = self.fc_type(out)

        # PN: 극성
        out_pn = self.fc_pn(out)

        # TIME: 시제
        out_time = self.fc_time(out)

        # SURE: 확실성 - Binary Classification
        out_sure = self.fc_sure(out)
        
        outputs = {'type': out_type, 'pn': out_pn, 'time': out_time, 'sure': out_sure}

        return outputs  
