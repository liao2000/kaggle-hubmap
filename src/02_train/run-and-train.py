# -*- coding: utf-8 -*-
# This file combines `train_02.py` and `run.py`; thus we can run multiple versions simultaneously.
# Please override `config`

import sys
sys.path.insert(0, '../')
import warnings
warnings.simplefilter('ignore')

from get_fold_idxs_list import get_fold_idxs_list
import pickle

import time
import pandas as pd
import numpy as np
import gc
import os
from os.path import join as opj
import pickle
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import HuBMAPDatasetTrain
from models import build_model
from scheduler import CosineLR
from utils import elapsed_time, fix_seed
from lovasz_loss import lovasz_hinge
from losses import criterion_lovasz_hinge_non_empty
from metrics import dice_sum, dice_sum_2
from get_config import get_config
config = get_config()

# override config
config['OUTPUT_PATH'] = './result_v0/02/'
config['model_name'] = 'seresnext101'
config['pretrain_path_list'] = None
output_path = config['OUTPUT_PATH']
fold_list = config['FOLD_LIST']
pretrain_path_list = config['pretrain_path_list']
device = config['device']

def run(seed, data_df, pseudo_df, trn_idxs_list, val_idxs_list):
    log_cols = ['fold', 'epoch', 'lr',
                'loss_trn', 'loss_val',
                'trn_score', 'val_score', 
                'elapsed_time']
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    criterion_clf = nn.BCEWithLogitsLoss().to(device)
    
    for fold, (trn_idxs, val_idxs) in enumerate(zip(trn_idxs_list, val_idxs_list)):
        if fold in fold_list:
            pass
        else:
            continue
        
        print('seed = {}, fold = {}'.format(seed, fold))
        
        log_df = pd.DataFrame(columns=log_cols, dtype=object)
        log_counter = 0

        #dataset
        trn_df = data_df.iloc[trn_idxs].reset_index(drop=True)
        val_df = data_df.iloc[val_idxs].reset_index(drop=True)
        
        #add pseudo label
        if pseudo_df is not None:
            trn_df = pd.concat([trn_df, pseudo_df], axis=0).reset_index(drop=True)
        
        # dataloader
        valid_dataset = HuBMAPDatasetTrain(val_df, config, mode='valid')
        valid_loader  = DataLoader(valid_dataset, batch_size=config['test_batch_size'],
                                   shuffle=False, num_workers=4, pin_memory=True)
        
        #model
        model = build_model(model_name=config['model_name'],
                            resolution=config['resolution'], 
                            deepsupervision=config['deepsupervision'], 
                            clfhead=config['clfhead'],
                            clf_threshold=config['clf_threshold'],
                            load_weights=True).to(device, torch.float32)

    
        if pretrain_path_list is not None:
            model.load_state_dict(torch.load(pretrain_path_list[fold]), strict=False)

        optimizer = optim.Adam(model.parameters(), **config['Adam'])
        scaler = torch.cuda.amp.GradScaler()
        
        if config['lr_scheduler_name']=='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_scheduler']['ReduceLROnPlateau'])
        elif config['lr_scheduler_name']=='CosineAnnealingLR':
            scheduler = CosineLR(optimizer, **config['lr_scheduler']['CosineAnnealingLR'])
        elif config['lr_scheduler_name']=='OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader),
                                                      **config['lr_scheduler']['OneCycleLR'])
        
        #training
        val_score_best  = -1e+99
        val_score_best2 = -1e+99
        loss_val_best   = 1e+99
        epoch_best = 0
        counter_ES = 0
        trn_score = 0
        trn_score_each = 0
        start_time = time.time()
        for epoch in range(1, config['num_epochs']+1):
            if epoch < config['restart_epoch_list'][fold]:
                scheduler.step()
                continue

            print('lr : ', [ group['lr'] for group in optimizer.param_groups ])
            
            #train
            trn_df['binned'] = trn_df['binned'].apply(lambda x:config['binned_max'] if x>=config['binned_max'] else x)
            n_sample = trn_df['is_masked'].value_counts().min()
            trn_df_0 = trn_df[trn_df['is_masked']==False].sample(n_sample, replace=True)
            trn_df_1 = trn_df[trn_df['is_masked']==True].sample(n_sample, replace=True)
            
            n_bin = int(trn_df_1['binned'].value_counts().mean())
            trn_df_list = []
            for bin_size in trn_df_1['binned'].unique():
                trn_df_list.append(trn_df_1[trn_df_1['binned']==bin_size].sample(n_bin, replace=True))
            trn_df_1 = pd.concat(trn_df_list, axis=0)
            trn_df_balanced = pd.concat([trn_df_1, trn_df_0], axis=0).reset_index(drop=True)
            train_dataset = HuBMAPDatasetTrain(trn_df_balanced, config, mode='train')
            train_loader  = DataLoader(train_dataset, batch_size=config['trn_batch_size'],
                                       shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
            model.train()
            running_loss_trn = 0
            trn_score_numer = 0
            trn_score_denom = 0
            y_preds = []
            y_trues = []
            counter = 0
            tk0 = tqdm(train_loader, total=int(len(train_loader)))
            for i,data in enumerate(tk0):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    batch,c,h,w = data['img'].shape
                    if config['clfhead']:
                        y_clf = data['label'].to(device, torch.float32, non_blocking=True)
                        if config['deepsupervision']:
                            logits,logits_deeps,logits_clf = model(data['img'].to(device, torch.float32, non_blocking=True))
                        else:
                            logits,logits_clf = model(data['img'].to(device, torch.float32, non_blocking=True))
                    else:
                        if config['deepsupervision']:
                            logits,logits_deeps = model(data['img'].to(device, torch.float32, non_blocking=True))
                        else:
                            logits = model(data['img'].to(device, torch.float32, non_blocking=True))
                    y_true = data['mask'].to(device, torch.float32, non_blocking=True)
                    dice_numer, dice_denom = dice_sum_2((torch.sigmoid(logits)).detach().cpu().numpy(), 
                                                        y_true.detach().cpu().numpy(), 
                                                        dice_threshold=config['dice_threshold'])
                    trn_score_numer += dice_numer 
                    trn_score_denom += dice_denom
                    loss = criterion(logits,y_true)
                    loss += lovasz_hinge(logits.view(-1,h,w), y_true.view(-1,h,w))
                    if config['deepsupervision']:
                        for logits_deep in logits_deeps:
                            loss += 0.1 * criterion_lovasz_hinge_non_empty(criterion, logits_deep, y_true)
                    if config['clfhead']:
                        loss += criterion_clf(logits_clf.squeeze(-1),y_clf)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #loss.backward()
                #optimizer.step()
                if config['lr_scheduler_name']=='OneCycleLR':
                    scheduler.step()
                running_loss_trn += loss.item() * batch
                counter  += 1
                tk0.set_postfix(loss=(running_loss_trn / (counter * train_loader.batch_size) ))
            epoch_loss_trn = running_loss_trn / len(train_dataset)
            trn_score = trn_score_numer / trn_score_denom
            
            #release GPU memory cache
            del data, loss,logits,y_true
            torch.cuda.empty_cache()
            gc.collect()

            #eval
            model.eval()
            loss_val  = 0
            val_score_numer = 0
            val_score_denom = 0
            y_preds = []
            y_trues = []
            tk1 = tqdm(valid_loader, total=int(len(valid_loader)))
            for i,data in enumerate(tk1):
                with torch.no_grad():
                    batch,c,h,w  = data['img'].shape
                    if config['clfhead']:
                        y_clf = data['label'].to(device, torch.float32, non_blocking=True)
                        if config['deepsupervision']:
                            logits,logits_deeps,logits_clf = model(data['img'].to(device, torch.float32, non_blocking=True))
                        else:
                            logits,logits_clf = model(data['img'].to(device, torch.float32, non_blocking=True))
                    else:
                        if config['deepsupervision']:
                            logits,logits_deeps = model(data['img'].to(device, torch.float32, non_blocking=True))
                        else:
                            logits = model(data['img'].to(device, torch.float32, non_blocking=True))
                    y_true = data['mask'].to(device, torch.float32, non_blocking=True)
                    dice_numer, dice_denom = dice_sum_2((torch.sigmoid(logits)).detach().cpu().numpy(), 
                                                        y_true.detach().cpu().numpy(), 
                                                        dice_threshold=config['dice_threshold'])
                    val_score_numer += dice_numer 
                    val_score_denom += dice_denom
                    loss_val += criterion(logits,y_true).item() * batch
                    loss_val += lovasz_hinge(logits.view(-1,h,w), y_true.view(-1,h,w)).item() * batch
                    if config['deepsupervision']:
                        for logits_deep in logits_deeps:
                            loss_val += 0.1 * criterion_lovasz_hinge_non_empty(criterion, logits_deep, y_true).item() * batch
                    if config['clfhead']:
                        loss_val += criterion_clf(logits_clf.squeeze(-1), y_clf).item() * batch
                #release GPU memory cache
                del data,logits,y_true
                torch.cuda.empty_cache()
                gc.collect()
            loss_val  /= len(valid_dataset)
            val_score = val_score_numer / val_score_denom
            
            #logging
            log_df.loc[log_counter,log_cols] = np.array([fold, epoch,
                                                         [ group['lr'] for group in optimizer.param_groups ],
                                                         epoch_loss_trn, loss_val, 
                                                         trn_score, val_score,
                                                         elapsed_time(start_time)], dtype='object')
            log_counter += 1
            
            #monitering
            print('epoch {:.0f} loss_trn = {:.5f}, loss_val = {:.5f}, trn_score = {:.4f}, val_score = {:.4f}'.format(epoch, epoch_loss_trn, loss_val, trn_score, val_score))
            if epoch%10 == 0:
                print(' elapsed_time = {:.1f} min'.format((time.time() - start_time)/60))
                
            if config['early_stopping']:
                if loss_val < loss_val_best: #val_score > val_score_best:
                    val_score_best = val_score #update
                    loss_val_best  = loss_val #update
                    epoch_best     = epoch #update
                    counter_ES     = 0 #reset
#                    torch.save(model.state_dict(), output_path+f'model_seed{seed}_fold{fold}_bestloss.pth') #save
                    print('model (best loss) saved')
                else:
                    counter_ES += 1
                if counter_ES > config['patience']:
                    print('early stopping, epoch_best {:.0f}, loss_val_best {:.5f}, val_score_best {:.5f}'.format(epoch_best, loss_val_best, val_score_best))
                    break
            else:
#                torch.save(model.state_dict(), output_path+f'model_seed{seed}_fold{fold}_bestloss.pth') #save
                pass
            if val_score > val_score_best2:
                val_score_best2 = val_score #update
                torch.save(model.state_dict(), output_path+f'model_seed{seed}_fold{fold}_bestscore.pth') #save
                print('model (best score) saved')
            
            if config['lr_scheduler_name']=='ReduceLROnPlateau':
                scheduler.step(loss_val)
                #scheduler.step(val_score)
            elif config['lr_scheduler_name']=='CosineAnnealingLR':
                scheduler.step()
                
            #for snapshot ensemble
            if config['lr_scheduler_name']=='CosineAnnealingLR':
                t0 = config['lr_scheduler']['CosineAnnealingLR']['t0']
                if (epoch%(t0+1)==0) or (epoch%(t0)==0) or (epoch%(t0-1)==0):
                    #torch.save(model.state_dict(), output_path+f'model_seed{seed}_fold{fold}_epoch{epoch}.pth') #save
                    print(f'model saved epoch{epoch} for snapshot ensemble')
            
            #save result
            log_df.to_csv(output_path+f'log_seed{seed}_fold{fold}.csv', index=False)

            print('')
            
        #best model
        if config['early_stopping']&(counter_ES<=config['patience']):
            print('epoch_best {:d}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(epoch_best, loss_val_best, val_score_best))
        
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print('')


if __name__=='__main__':
    # config
    fix_seed(2021)
    FOLD_LIST = config['FOLD_LIST']
    VERSION = config['VERSION']
    INPUT_PATH = config['INPUT_PATH']
    OUTPUT_PATH = config['OUTPUT_PATH']
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = config['device']
    print(device)
    
    # import data
    train_df = pd.read_csv(opj(INPUT_PATH, 'train.csv'))
    info_df  = pd.read_csv(opj(INPUT_PATH,'HuBMAP-20-dataset_information.csv'))
    sub_df = pd.read_csv(opj(INPUT_PATH, 'sample_submission.csv'))
    print('train_df.shape = ', train_df.shape)
    print('info_df.shape  = ', info_df.shape)
    print('sub_df.shape = ', sub_df.shape)
    
    # dataset
    data_df = []
    for data_path in config['train_data_path_list']:
        _data_df = pd.read_csv(opj(data_path,'data.csv'))
        _data_df['data_path'] = data_path
        data_df.append(_data_df)
    data_df = pd.concat(data_df, axis=0).reset_index(drop=True)

    print('data_df.shape = ', data_df.shape)
    data_df = data_df[data_df['std_img']>10].reset_index(drop=True)
    print('data_df.shape = ', data_df.shape)
    data_df['binned'] = np.round(data_df['ratio_masked_area'] * config['multiplier_bin']).astype(int)
    data_df['is_masked'] = data_df['binned']>0

    trn_df = data_df.copy()
    trn_df['binned'] = trn_df['binned'].apply(lambda x:config['binned_max'] if x>=config['binned_max'] else x)
    trn_df_1 = trn_df[trn_df['is_masked']==True]
    print(trn_df['is_masked'].value_counts())
    print(trn_df_1['binned'].value_counts())
    print('mean = ', int(trn_df_1['binned'].value_counts().mean()))
    
    info_df['image_name'] = info_df['image_file'].apply(lambda x:x.split('.')[0])
    patient_mapper = {}
    for (x,y) in info_df[['image_name','patient_number']].values:
        patient_mapper[x] = y
    data_df['patient_number'] = data_df['filename_img'].apply(lambda x:patient_mapper[x.split('_')[0]])
    
    val_patient_numbers_list = [
        [63921], # fold0
        [68250], # fold1
        [65631], # fold2
        [67177], # fold3
    ]
    
    # train
    for seed in config['split_seed_list']:
        trn_idxs_list, val_idxs_list = get_fold_idxs_list(data_df, val_patient_numbers_list)
        with open(opj(config['OUTPUT_PATH'],f'trn_idxs_list_seed{seed}'), 'wb') as f:
            pickle.dump(trn_idxs_list, f)
        with open(opj(config['OUTPUT_PATH'],f'val_idxs_list_seed{seed}'), 'wb') as f:
            pickle.dump(val_idxs_list, f)
        run(seed, data_df, None, trn_idxs_list, val_idxs_list)
        
    # score
    score_list  = []
    for seed in config['split_seed_list']:
        for fold in config['FOLD_LIST']:
            log_df = pd.read_csv(opj(config['OUTPUT_PATH'],f'log_seed{seed}_fold{fold}.csv'))
            score_list.append(log_df['val_score'].max())
    print('CV={:.4f}'.format(sum(score_list)/len(score_list)))


