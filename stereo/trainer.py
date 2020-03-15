#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
#import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader import preprocess

import logging
logger = logging.getLogger(__name__)


visualize_disp = False # True # 


class StereoTrainer(nn.Module):
    
    def __init__(self, model, lossfun):
        super(StereoTrainer, self).__init__()

        self.model = model
        self.lossfun = lossfun
        self.flag_ec = lossfun.flag_ec
        

    def forward(self, batch):

        if(self.lossfun.flag_unsupervised_LR):
            return self.loss_unsupervised_LR(batch)
        elif(self.lossfun.flag_unsupervised_L):
            return self.loss_unsupervised_L(batch, self.flag_ec)
        else:
            return self.loss_supervised(batch)


    def preprocess(self, imgL, imgR):

        process = [preprocess.augment_color(True) for i in range(imgL.size(0))]
        imgL = torch.cat(list(map(lambda f, x: f(x)[None], process, imgL)), dim=0)
        imgR = torch.cat(list(map(lambda f, x: f(x)[None], process, imgR)), dim=0)
        return imgL, imgR


    def loss_supervised(self, batch):
       
        assert 3 <= len(batch), str([t.shape for t in batch])
        imL, imR, dispL = batch
        
        argst = {'scales': self.model.scales, }
        loss_name = self.lossfun.name
        
        with torch.autograd.set_detect_anomaly(True):

            imL, imR = self.preprocess(imL, imR)
            out = self.model(imL, imR)

            # compute loss
            if('sl1' in loss_name) and ('ce' in loss_name):
                loss_ex, disps, similars = out
                step_disps = self.model.step_disps
                argst.update({'step_disps':step_disps, 'similars':similars, 'disps':disps, })

            elif('sl1' in loss_name):
                loss_ex, disps = out
                argst.update({'disps':disps, })

            elif('ce' in loss_name):
                loss_ex, similars = out
                step_disps = self.model.step_disps
                argst.update({'step_disps':step_disps, 'similars':similars, })

            else:
                raise Exception('no supported lossfun')

            argst.update({'disp_true': dispL})
            loss = self.lossfun(argst)
            if(loss_ex > 0): loss = loss + loss_ex.mean()

        # visualize images
        if(visualize_disp):
            
            row, col = 3, 3
            unnormalize = preprocess.unnormalize_imagenet()
            to_numpy = lambda tensor: tensor.cpu().data.numpy()

            plt.subplot(row, col, 1); plt.imshow(to_numpy(unnormalize(imL[0])).transpose(1, 2, 0)) 
            plt.subplot(row, col, 2); plt.imshow(to_numpy(unnormalize(imR[0])).transpose(1, 2, 0)) 
            plt.subplot(row, col, 3); plt.imshow(to_numpy(dispL[0, 0]))
            
            for i in range(len(disps)):
                if(disps[i] is None): continue
                if isinstance(disps[i], list): disps[i] = torch.cat(disps[i], dim=-2)
                plt.subplot(row, col, 3+i); plt.imshow(to_numpy(disps[i][0, 0]))
            plt.show()

        # return
        return loss[None]


    def loss_unsupervised_LR(self, batch):
       
        assert 2 <= len(batch), str([t.shape for t in batch])
        imL1, imR1 = batch[:2]
        bn, c, h, w = imL1.shape
        
        nedge = self.lossfun.nedge
        assert 2*nedge < w

        imL2 = torch.flip(imR1, dims=[-1])
        imR2 = torch.flip(imL1, dims=[-1])
        
        imL = torch.cat([imL1, imL2], dim=0)
        imR = torch.cat([imR1, imR2], dim=0)

        # color augmentation 
        imL_aug, imR_aug = self.preprocess(imL[..., nedge:w-nedge], imR[..., nedge:w-nedge])
        
        kedge = 2*nedge/float(w)
        rect={'xs': -1+kedge, 'xe':1-kedge, 'ys':-1, 'ye':1}
        argst = {'scales': self.model.scales, }

        with torch.autograd.set_detect_anomaly(True):

            # compute output
            loss_ex, dispLs = self.model(imL_aug, imR_aug)
            dispLs1, dispLs2 = [], []
            for disps in dispLs:
                if(isinstance(disps, list)):
                    tdispLs1, tdispLs2 = [], []
                    for tdisps in disps:
                        tdispLs1.append(tdisps[:bn])
                        tdispLs2.append(tdisps[bn:])
                    dispLs1.append(tdispLs1)
                    dispLs2.append(tdispLs2)
                else:
                    dispLs1.append(disps[:bn])
                    dispLs2.append(disps[bn:])

            # compute loss
            imL1 = imL1[..., nedge:w-nedge]
            imL2 = imL2[..., nedge:w-nedge]
            argst.update({
                    "imR1": imR1, "imL1": imL1, "dispLs1": dispLs1, "rect1": rect, 
                    "imR2": imR2, "imL2": imL2, "dispLs2": dispLs2, "rect2": rect, 
                    })
            loss = self.lossfun(argst)
            if(loss_ex>0): loss += loss_ex.mean()

        # visualize images
        if(visualize_disp):
            
            row, col = 4, 4
            unnormalize = preprocess.unnormalize_imagenet()
            to_numpy = lambda tensor: tensor.cpu().data.numpy()

            imgs = [imL1, imR1, imL2, imR2, imL_aug[:bn], imR_aug[:bn], imL_aug[bn:], imR_aug[bn:]]
            for i in range(4, len(imgs)):
                imgs[i][:1] = unnormalize(imgs[i][:1].data)
            for i in range(len(imgs)):
                plt.subplot(row, col, 1+i); plt.imshow(to_numpy(imgs[i][0]).transpose(1, 2, 0)) 

            for i in range(len(dispLs)):
                if(dispLs[i] is None): continue
                if isinstance(dispLs[i], list): dispLs[i] = torch.cat(dispLs[i], dim=-2)
                plt.subplot(row, col, 9+i); plt.imshow(to_numpy(dispLs[i][0, 0]))
            plt.show()

        # return
        return loss[None]


    def loss_unsupervised_L(self, batch, flag_edge_con=False):
       
        assert 2 <= len(batch), str([t.shape for t in batch])
        imL1, imR1 = batch[:2]
        bn, c, h, w = imL1.shape
        
        nedge = self.lossfun.nedge
        assert 2*nedge < w

        # color augmentation 
        imL1_aug, imR1_aug = self.preprocess(imL1[..., nedge:w-nedge], imR1[..., nedge:w-nedge])
        
        kedge = 2*nedge/float(w)
        rect={'xs': -1+kedge, 'xe':1-kedge, 'ys':-1, 'ye':1}
        argst = {'scales': self.model.scales, }

        with torch.autograd.set_detect_anomaly(True):

            # compute output
            output = self.model(imL1_aug, imR1_aug)

            # compute loss
            imL1 = imL1[..., nedge:w-nedge]

            if(flag_edge_con):
                loss_ex, dispLs1, edges1, cons1 = output
                argst.update({
                        "imR1": imR1, "imL1": imL1, "dispLs1": dispLs1, 
                        "edges1": edges1, "cons1": cons1, "rect1": rect, 
                        })

            else:
                loss_ex, dispLs1 = output
                argst.update({"imR1": imR1, "imL1": imL1, "dispLs1": dispLs1, "rect1": rect, })

            loss = self.lossfun(argst)
            if(loss_ex>0): loss += loss_ex.mean()

        # visualize images
        if(visualize_disp):
            
            row, col = 4, 4
            unnormalize = preprocess.unnormalize_imagenet()
            to_numpy = lambda tensor: tensor.cpu().data.numpy()

            imgs = [imL1, imR1, imL1_aug, imR1_aug]
            for i in range(4, len(imgs)):
                imgs[i][:1] = unnormalize(imgs[i][:1].data)
            for i in range(len(imgs)):
                plt.subplot(row, col, 1+i); plt.imshow(to_numpy(imgs[i][0]).transpose(1, 2, 0)) 

            for i in range(len(dispLs1)):
                if(dispLs1[i] is None): continue
                if isinstance(dispLs1[i], list): dispLs1[i] = torch.cat(dispLs1[i], dim=-2)
                plt.subplot(row, col, 9+i); plt.imshow(to_numpy(dispLs1[i][0, 0]))
            plt.show()

        # return
        return loss[None]



