#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import utils

import logging
logger = logging.getLogger(__name__)

visualize_disp = False # True # 


class StereoLoss(nn.Module):
    '''
    supported lossfun as follow:
    
    (1)supervised mode [with flag "SV-(SL1/CE/SL1+CE)"]: 
        SL1: smooth L1 loss
        CE : subpixel cross entropy loss
        SL1+CE : combine SL1 and CE
    
    (2)unsupervised mode with left and right view [with flag "DUSV-(A[S(1/2/3/4)]C(1/2)[-AD][-M]"]: 
        
        A: reconstruction loss (0.85*(1-ssim) + 0.15*im_delt)
        
        S[1/2]: disparity smooth loss with first/second order difference
        S3: disparity smooth loss with normal second order difference
        S4: disparity smooth loss with local plane
        
        C1: left-right consistency loss with disp_delt
        C2: left-right consistency loss with double wraped image

        [-AD]: weight = torch.exp(-im_diff/0.5*mean(im_diff))
        [-M]: compute loss_ap and loss_lr only on pixels within left and right image

    (3)unsupervised mode with left view [with flag "LUSV-(A[S(1/2/3/4)][-AD])/(AS(1/2/3)-EC)"]: 
        
        
        A: reconstruction loss (0.85*(1-ssim) + 0.15*im_delt)
        
        S[1/2]: disparity smooth loss with first/second order difference
        S3: disparity smooth loss with normal second order difference
        S4: disparity smooth loss with local plane
        
        [-AD]: weight = torch.exp(-im_diff/0.5*mean(im_diff))
        [-EC]: predict edge and consistency from model
    '''

    def __init__(self, args):
        super(StereoLoss, self).__init__()

        self.weight_ap = 1.0
        self.weight_ds = 0.01
        self.weight_lr = 0.01
        self.weight_dm = 0.0001
        self.weight_similary = lambda s: 0.01 + max(0, s-0.75)*0.5
        self.weights = [1, 0.7, 0.5, 0.5] + [0.3]*6
        
        # args [maxdisp, name, start, tname]
        self.maxdisp = args.maxdisp
        self.starts = [t.lower() for t in ['SV', 'DUSV', 'LUSV']]
        self.name = args.loss_name.lower()
        idx_start = self.name.find('-')
        assert 0 < idx_start
        self.start = self.name[:idx_start]
        self.tname = self.name.split('-')[1]
        assert (self.start in self.starts) , 'unspported lossfun[ %s ]' % self.name
       
        # args [flag_supervised, flag_unsupervised_LR, flag_unsupervised_L, flag_ad, flag_mask, flag_ec]
        self.flag_supervised = ('sv' == self.start)
        self.flag_unsupervised_LR = ('dusv' == self.start)
        self.flag_unsupervised_L = ('lusv' == self.start)
        self.flag_ds = ('as' in self.tname)
        self.flag_lr = ('c' in self.tname)
        self.flag_ad = ('-ad' in self.name)
        self.flag_mask = ('-m' in self.name)
        self.flag_ec = ('-ec' in self.name)
        self.nedge = args.nedge  if(not self.flag_supervised)and(self.flag_mask) else 0

        msg = 'kargs of lossfun[%s] as follow: \n' % self.name
        msg += ' tname: '
        msg += str([self.tname.upper()]) + '\n'
        msg += ' flag_supervised, flag_unsupervised_LR, flag_unsupervised_L: '
        msg += str([self.flag_supervised, self.flag_unsupervised_LR, self.flag_unsupervised_L]) + '\n'
        msg += ' flag_ds, flag_lr: '
        msg += str([self.flag_ds, self.flag_lr]) + '\n'
        msg += ' flag_ad, flag_mask, flag_ec: '
        msg += str([self.flag_ad, self.flag_mask, self.flag_ec]) + '\n'
        msg += ' nedge: '
        msg += str([self.nedge]) + '\n'
        logger.info(msg)

        # args [flag_FCTF, mode_down_disp, mode_down_img]
        self.flag_FCTF = args.flag_FCTF
        self.mode_down_disp = args.mode_down_disp.lower() # avg/max
        self.mode_down_img = args.mode_down_img.lower() # Simple/Gaussion/DoG

        # ssim function
        self.ssim = utils.SSIM(kernel_size=11, sigma=1, padding=5)

        # downsample for disp and image
        self.down_disp2x2 = utils.AvgPool2d_mask(2, ceil_mode=True) \
                            if 'avg' == self.mode_down_disp else \
                            nn.MaxPool2d(2, ceil_mode=True)
        self.down_img2x2 = utils.GaussianBlur2d_linear(kernel_size=7, sigma=1.0, stride=2, padding=3)
        self.pyramid_gaussion = utils.GaussionPyramid(octaves=1, noctave=4, sigma=1.0, noctave_ex=0)
        self.pyramid_DoG = utils.DoGPyramid(octaves=1, noctave=4, sigma=1.0, noctave_ex=0)

        # gaussion function
        self.blur_gaussion = utils.GaussianBlur2d_linear(kernel_size=7, sigma=1.0, stride=1, padding=3)


    # ------------------------------ loss_pyramid_supervised ---------------------------------
    def _disp_pyramid(self, disp, levels):

        tmaxdisp = self.maxdisp
        fun_mask = lambda disp, maxD: (disp > 0) & (disp < maxD)
        disps = [disp]
        masks = [fun_mask(disp, tmaxdisp)]
        for i in range(levels-1):
            tmaxdisp /= 2.0
            disps.append(self.down_disp2x2(disps[-1])/2.0)
            masks.append(fun_mask(disps[-1], tmaxdisp))
        return disps, masks
    
    
    def _visualize_disp(self, mask, disp, disp_true, threshold, tloss, scale):
        
        to_numpy = lambda tensor: tensor.cpu().data.numpy()
        
        n = 1
        pad = max(1, disp.size(-1)//100)
        
        imgs = torch.cat([disp[:n], disp_true[:n]], dim=0)
        imgs = make_grid(imgs, nrow=max(2, n), padding=pad, normalize=False)

        plt.subplot(211); plt.imshow(to_numpy(imgs[0]))
        plt.title('scale=%d, tloss=%.2f, threshold=%.2f' % (scale, tloss, threshold))
        
        imgs = (disp[:n]-disp_true[:n]).abs()
        imgs[~mask[:n]] = 0
        imgs = torch.cat([imgs.clamp(0, threshold), imgs.clamp(0, 3)/3], dim=0)
        imgs = make_grid(imgs, nrow=max(2, n), padding=pad, normalize=False)

        plt.subplot(212); plt.imshow(to_numpy(imgs[0]))
        plt.show()
    

    def _losses_SL1(self, disp_true, scales, disps):

        # create pyramid of disps_true
        ndim = disp_true.dim()
        disps_true, masks = self._disp_pyramid(disp_true, max(scales)+1)
        
        # accumlate loss
        loss = torch.zeros(1).type_as(disp_true)
        threshold = 1
        
        for level, tdisps in zip(scales, disps):

            # accumlate loss on scale of level
            flag_break = False
            tdisp_true, tmask = disps_true[level], masks[level]
            if(0 == len(tmask[tmask])):
                continue
            
            if(not isinstance(tdisps, list)):
                if(tdisps.dim() != ndim):
                    continue
                tdisps = [tdisps]
            idx_max = len(tdisps)-1

            for idx, tdisp in enumerate(tdisps):

                tloss = utils.loss_disp(tdisp, tdisp_true, tmask)
                loss = loss + tloss*self.weights[idx_max-idx]

                if(visualize_disp): 
                    self._visualize_disp(tmask, tdisp, tdisp_true, threshold, tloss, level)

                if self.flag_FCTF and (tloss>threshold): 
                    flag_break = True; break

            if(flag_break): break

        return loss


    def _losses_CE(self, disp_true, scales, similars, step_disps):

        # create pyramid of disps_true
        ndim = disp_true.dim() + 1
        disps_true, masks = self._disp_pyramid(disp_true, max(scales)+1)
        
        # accumlate loss
        loss = torch.zeros(1).type_as(disp_true)
        threshold = 1
        
        for level, step_disp, tsimilars in zip(scales, step_disps, similars):
            
            # accumlate loss on scale of level
            flag_break = False
            tdisp_true, tmask = disps_true[level], masks[level]
            if(0 == len(tmask[tmask])):
                continue
            
            if(not isinstance(tsimilars, list)):
                if(tsimilars.dim() != ndim):
                    continue
                tsimilars = [tsimilars]
            idx_max = len(tsimilars)-1

            for idx, tsimilar in enumerate(tsimilars):

                tloss = utils.loss_subpixel_cross_entropy(tsimilar, tdisp_true[:, 0], step_disp, mask=tmask[:, 0])
                loss = loss + tloss*self.weights[idx_max-idx]

                if(visualize_disp):
                    _, tdisp = tsimilar.max(dim=1, keepdim=True)
                    self._visualize_disp(tmask, tdisp.float()*step_disp, tdisp_true, threshold, tloss, level)

                if self.flag_FCTF and (tloss>threshold): 
                    flag_break = True; break

            if(flag_break): break

        return loss


    def _losses_CE_SL1(self, disp_true, scales, disps, similars, step_disps):

        # create pyramid of disps_true
        ndim = disp_true.dim() + 1
        disps_true, masks = self._disp_pyramid(disp_true, max(scales)+1)
        
        # accumlate loss
        loss = torch.zeros(1).type_as(disp_true)
        threshold = 1
        
        for level, step_disp, tsimilars, tdisps in zip(scales, step_disps, similars, disps):
            
            # accumlate loss on scale of level
            flag_break = False
            tdisp_true, tmask = disps_true[level], masks[level]
            if(0 == len(tmask[tmask])):
                continue
            
            if(not isinstance(tdisps, list)):
                if(tdisps.dim() != ndim):
                    continue
                tdisps = [tdisps]
                tsimilars = [tsimilars]
            idx_max = len(tdisps)-1

            for idx, (tsimilar, tdisp) in enumerate(zip(tsimilars, tdisps)):

                tloss_ce = utils.loss_subpixel_cross_entropy(tsimilar, tdisp_true[:, 0], step_disp, mask=tmask[:, 0])
                tloss_sl1 = utils.loss_disp(tdisp, tdisp_true, tmask)
                tloss = tloss_ce + 0.2*tloss_sl1
                loss = loss + tloss*self.weights[idx_max-idx]

                if(visualize_disp):
                    self._visualize_disp(tmask, tdisp, tdisp_true, threshold, tloss, level)

                if self.flag_FCTF and (tloss_sl1>threshold): 
                    flag_break = True; break

            if(flag_break): break

        return loss


    # ------------------------------ loss_unsupervised ---------------------------------
    def _img_pyramid(self, img, levels):
        
        if('dog' == self.mode_down_img):
            return self.pyramid_DoG(img, levels, normalize=True)
        elif('gaussion' == self.mode_down_img):
            return self.pyramid_gaussion(img, levels, normalize=True)
        else:
            img_pyramid = [img]
            # pyramid
            for i in range(1, levels):
                img_pyramid.append(self.down_img2x2(img_pyramid[-1]))
            return img_pyramid


    def _img_blur(self, img):
        return self.blur_gaussion(img)

        
    def _loss_ASnC1(self, im, im_wrap, disp, disp_wrap=None, weight_common=None):

        # --------------- im_ssim and mean similary --------------------
        mask_ap = (im_wrap.abs().sum(dim=1, keepdim=True) != 0)
        im_ssim = self.ssim(im, im_wrap).mean(dim=1, keepdim=True)
        similary = im_ssim[mask_ap].mean().item()

        # ---------------- set weight_ds and weight_lr ------------------
        self.weight_ds = self.weight_similary(similary)
        self.weight_lr = self.weight_ds
        
        # ------------------------ loss_ap -----------------------------
        im_L1 = torch.abs(im - im_wrap).mean(dim=1, keepdim=True)
        loss_ap = (0.85*0.5)*(1 - im_ssim) + 0.15*im_L1

        # ------------------------ loss_lr -----------------------------
        loss_lr = None if(disp_wrap is None) else (disp - disp_wrap).abs()

        # ----- adjust loss_ap and loss_lr with weight_common ----------
        if(disp_wrap is not None) and (weight_common is not None):

            mask_occ = (disp_wrap==0)

            weight_ap = weight_common.clone()
            weight_ap[mask_occ & mask_ap] = 1.0
            loss_ap = loss_ap * weight_ap

            tweight_lr = weight_common.clone()
            tweight_lr[mask_occ] = 0.0
            loss_lr = loss_lr * tweight_lr

        # ----------------------loss_all--------------------------------
        loss = loss_ap.mean()*self.weight_ap 

        if self.flag_lr and (loss_lr is not None):
            loss += loss_lr.mean()*self.weight_lr
        
        if(self.flag_ds):
            fun_ds = utils.loss_disp_smooth
            mode = self.tname[1:3]
            loss_ds = fun_ds(im, disp, mode, self.flag_ad)
            loss += loss_ds.mean()*self.weight_ds

        return loss, similary


    def _loss_ASnC2(self, im, im_wrap, im_wrap1, disp, weight_common=None):

        # --------------- im_ssim and mean similary --------------------
        mask_ap = (im_wrap.abs().sum(dim=1, keepdim=True) != 0)
        im_ssim = self.ssim(im, im_wrap).mean(dim=1, keepdim=True)
        similary = im_ssim[mask_ap].mean().item()

        # ---------------- set weight_ds and weight_lr ------------------
        self.weight_ds = self.weight_similary(similary)
        self.weight_lr = self.weight_ap
        
        # ------------------------ loss_ap -----------------------------
        im_L1 = torch.abs(im - im_wrap).mean(dim=1, keepdim=True)
        loss_ap = (0.85*0.5)*(1 - im_ssim) + 0.15*im_L1

        # ------------------------ loss_lr -----------------------------
        loss_lr = (im - im_wrap1).abs().mean(dim=1, keepdim=True)

        # ----- adjust loss_ap and loss_lr with weight_common ----------
        if(im_wrap1 is not None) and (weight_common is not None):

            mask_occ = (im_wrap1.abs().sum(dim=1, keepdim=True) == 0)

            weight_ap = weight_common.clone()
            weight_ap[mask_occ & mask_ap] = 1.0
            loss_ap = loss_ap * weight_ap

            tweight_lr = weight_common.clone()
            tweight_lr[mask_occ] = 0.0
            loss_lr = loss_lr * tweight_lr

        # ----------------------loss_all--------------------------------
        loss = loss_ap.mean()*self.weight_ap

        if self.flag_lr:
            loss += loss_lr.mean()*self.weight_lr
        
        if(self.flag_ds):
            fun_ds = utils.loss_disp_smooth
            mode = self.tname[1:3]
            loss_ds = fun_ds(im, disp, mode, self.flag_ad)
            loss += loss_ds.mean()*self.weight_ds

        return loss, similary


    def _loss_edge_con(self, im, im_wrap, disp, edge, consistency):

        # ----------------img_ssim and mean similary---------------------
        img_ssim = self.ssim(im, im_wrap).mean(dim=1, keepdim=True)
        if(4 > consistency.dim()):
            similary = img_ssim.mean().item()
        else:
            similary = ((img_ssim*consistency).mean()/consistency.mean().clamp(0.1)).item()
        self.weight_ds = self.weight_similary(similary)
        
        # ----------------set loss_ap and loss_lr---------------------
        loss_ap = (0.85*0.5)*(1 - img_ssim) + 0.15*torch.abs(im - im_wrap).mean(dim=1, keepdim=True)

        # ----------------------loss_all----------------------------
        if(4 > consistency.dim()):
            mask_ap = (im_wrap.abs().sum(dim=1, keepdim=True) != 0)
            loss_ap = (loss_ap*mask_ap.float()).mean()
        else:
            bn = disp.size(0)
            mean_inconsistency = (1-consistency.view(bn, -1).mean(-1)).view(bn, 1, 1, 1).clamp(0.01)
            loss_ap = (loss_ap*consistency).mean() + (mean_inconsistency*loss_ap).mean()

        mode = self.tname[1:3]
        loss_ds = utils.loss_disp_smooth_edge(disp, edge, mode)

        loss = loss_ap*self.weight_ap + loss_ds*self.weight_ds
        
        return loss, similary


    # ------------------------------ loss_pyramid_unsupervised ---------------------------------
    def _weight_common(self, disp, disp_wrap):

        disp_delt = torch.abs(disp - disp_wrap).detach()
        
        mask1 = disp_delt<1
        mask2 = (disp_delt<3) - mask1
        mask3 = disp_delt >= 3
        
        weight = torch.zeros_like(disp_delt)
        weight[mask1] = 1.0
        weight[mask2] = 1.0 - (disp_delt[mask2] - 1)*(0.99/2)
        weight[mask3] = 0.01
        
        msg = 'weight maxV: %f, minV: %f' % (weight.max().item(), weight.min().item())
        logger.debug(msg)

        return weight
    

    # losses for DUSV-ASnC1
    def _losses_ASnC1(self, scales, imR1, imL1, dispLs1, rect1, imR2, imL2, dispLs2, rect2):
        
        ndim = imL1.dim()

        uplevel = 3
        levels = max(scales) - uplevel + 1
        imLs1 = self._img_pyramid(imL1, levels)
        imLs2 = self._img_pyramid(imL2, levels)
        imRs1 = self._img_pyramid(imR1, levels)
        imRs2 = self._img_pyramid(imR2, levels)
        
        # compute loss
        loss = 0 
        threshold = 0.5
        
        kargs_up = {'mode': 'bilinear', 'align_corners': True}

        for level, tdispLs1, tdispLs2 in zip(scales, dispLs1, dispLs2):
            
            # accumlate loss on scale of level
            flag_break = False
            if(not isinstance(tdispLs1, list)):
                if(tdispLs1.dim() != ndim):
                    continue
                tdispLs1 = [tdispLs1]
                tdispLs2 = [tdispLs2]
            idx_max = len(tdispLs1)-1
            
            idx_level = max(level-uplevel, 0)
            factor = 2.0/(imRs1[idx_level].size(-1) - 1)
            for idx, (tdispL1, tdispL2) in enumerate(zip(tdispLs1, tdispLs2)):

                weight_common1, weight_common2 = None, None
            
                if(level > 0):
                    factor_up = 2**(level - idx_level)
                    h, w = imLs1[idx_level].shape[-2:]
                    tdispL1 = F.interpolate(tdispL1*factor_up, size=(h, w), **kargs_up)
                    tdispL2 = F.interpolate(tdispL2*factor_up, size=(h, w), **kargs_up)
                tdispL1_norm, tdispL2_norm = tdispL1*factor, tdispL2*factor

                imL_wrap1 = utils.imwrap(imRs1[idx_level], tdispL1_norm, rect1)
                imL_wrap2 = utils.imwrap(imRs2[idx_level], tdispL2_norm, rect2)
                
                dispL_wrap1 = utils.imwrap(tdispL2, -tdispL1_norm, rect={'xs': 1, 'xe':-1, 'ys':-1, 'ye':1})
                dispL_wrap2 = utils.imwrap(tdispL1, -tdispL2_norm, rect={'xs': 1, 'xe':-1, 'ys':-1, 'ye':1})
                
                if(self.flag_mask):
                    weight_common1 = self._weight_common(tdispL1, dispL_wrap1)
                    weight_common2 = self._weight_common(tdispL2, dispL_wrap2)
                
                tloss1, similary1 = self._loss_ASnC1(imLs1[idx_level], imL_wrap1, tdispL1, dispL_wrap1, weight_common=weight_common1)
                tloss2, similary2 = self._loss_ASnC1(imLs2[idx_level], imL_wrap2, tdispL2, dispL_wrap2, weight_common=weight_common2)
                tsimilary = 0.5*(similary1 + similary2)
                loss = loss + (tloss1 + tloss2)*self.weights[idx_max-idx]

                # imshow
                if(visualize_disp):
                    tensors = [imLs1[idx_level], imLs2[idx_level].flip([-1]), imL_wrap1, imL_wrap2.flip([-1]), 
                               tdispL1, tdispL2.flip([-1]), dispL_wrap1, dispL_wrap2.flip([-1])]
                    plt = utils.plot_tensors(*tensors)
                    plt.title('scale=%d, tsimilar=%.2f, threshold=%.2f' % (level, tsimilary, threshold))
                    plt.show()
        
                if self.flag_FCTF and (tsimilary < threshold): 
                    flag_break = True; break
            
            if(flag_break): break
        
        return loss
    
    # losses for DUSV-ASnC2
    def _losses_ASnC2(self, scales, imR1, imL1, dispLs1, rect1, imR2, imL2, dispLs2, rect2):
        
        ndim = imL1.dim()

        uplevel = 3
        levels = max(scales) - uplevel + 1
        imLs1 = self._img_pyramid(imL1, levels)
        imLs2 = self._img_pyramid(imL2, levels)
        imRs1 = self._img_pyramid(imR1, levels)
        imRs2 = self._img_pyramid(imR2, levels)
        
        # compute loss
        loss = 0 
        threshold = 0.5
        kargs_up = {'mode': 'bilinear', 'align_corners': True}

        for level, tdispLs1, tdispLs2 in zip(scales, dispLs1, dispLs2):
            
            # accumlate loss on scale of level
            flag_break = False
            if(not isinstance(tdispLs1, list)):
                if(tdispLs1.dim() != ndim):
                    continue
                tdispLs1 = [tdispLs1]
                tdispLs2 = [tdispLs2]
            idx_max = len(tdispLs1)-1
            
            idx_level = max(level-uplevel, 0)
            factor = 2.0/(imRs1[idx_level].size(-1) - 1)
            for idx, (tdispL1, tdispL2) in enumerate(zip(tdispLs1, tdispLs2)):

                weight_common1, weight_common2 = None, None
            
                if(level > 0):
                    factor_up = 2**(level - idx_level)
                    h, w = imLs1[idx_level].shape[-2:]
                    tdispL1 = F.interpolate(tdispL1*factor_up, size=(h, w), **kargs_up)
                    tdispL2 = F.interpolate(tdispL2*factor_up, size=(h, w), **kargs_up)
                tdispL1_norm, tdispL2_norm = tdispL1*factor, tdispL2*factor
                
                imL_wrap1 = utils.imwrap(imRs1[idx_level], tdispL1_norm, rect1)
                imL_wrap2 = utils.imwrap(imRs2[idx_level], tdispL2_norm, rect2)
                
                imL_wrap1_lr = utils.imwrap(imL_wrap2, -tdispL1_norm, rect={'xs': 1, 'xe':-1, 'ys':-1, 'ye':1})
                imL_wrap2_lr = utils.imwrap(imL_wrap1, -tdispL2_norm, rect={'xs': 1, 'xe':-1, 'ys':-1, 'ye':1})

                if(self.flag_mask):
                    dispL_wrap1 = utils.imwrap(tdispL2, -tdispL1_norm, rect={'xs': 1, 'xe':-1, 'ys':-1, 'ye':1})
                    dispL_wrap2 = utils.imwrap(tdispL1, -tdispL2_norm, rect={'xs': 1, 'xe':-1, 'ys':-1, 'ye':1})
                    weight_common1 = self._weight_common(tdispL1, dispL_wrap1)
                    weight_common2 = self._weight_common(tdispL2, dispL_wrap2)
                
                tloss1, similary1 = self._loss_ASnC2(imLs1[idx_level], imL_wrap1, imL_wrap1_lr, tdispL1, weight_common=weight_common1)
                tloss2, similary2 = self._loss_ASnC2(imLs2[idx_level], imL_wrap2, imL_wrap2_lr, tdispL2, weight_common=weight_common2)
                tsimilary = 0.5*(similary1 + similary2)
                loss = loss + (tloss1 + tloss2)*self.weights[idx_max-idx]

                # imshow
                if(visualize_disp):
                    tensors = [imLs1[idx_level], imLs2[idx_level].flip([-1]), imL_wrap1, imL_wrap2.flip([-1]), 
                               tdispL1, tdispL2.flip([-1]), imL_wrap1_lr, imL_wrap2_lr.flip([-1])]
                    plt = utils.plot_tensors(*tensors)
                    plt.title('scale=%d, tsimilar=%.2f, threshold=%.2f' % (level, tsimilary, threshold))
                    plt.show()
        
                if self.flag_FCTF and (tsimilary < threshold): 
                    flag_break = True; break
            
            if(flag_break): break
        
        return loss
    

    # losses for LUSV-ASn[-AD]
    def _losses_ASn(self, scales, imR1, imL1, dispLs1, rect1):
        
        ndim = imL1.dim()

        uplevel = 3
        levels = max(scales) - uplevel + 1
        imLs1 = self._img_pyramid(imL1, levels)
        imRs1 = self._img_pyramid(imR1, levels)
        
        # compute loss
        loss = 0 
        threshold = 0.5
        kargs_up = {'mode': 'bilinear', 'align_corners': True}

        for level, tdispLs1 in zip(scales, dispLs1):
            
            # accumlate loss on scale of level
            flag_break = False
            if(not isinstance(tdispLs1, list)):
                if(tdispLs1.dim() != ndim):
                    continue
                tdispLs1 = [tdispLs1]
            idx_max = len(tdispLs1)-1
            
            idx_level = max(level-uplevel, 0)
            factor = 2.0/(imRs1[idx_level].size(-1) - 1)
            for idx, tdispL1 in enumerate(tdispLs1):

                if(level > 0):
                    factor_up = 2**(level - idx_level)
                    h, w = imLs1[idx_level].shape[-2:]
                    tdispL1 = F.interpolate(tdispL1*factor_up, size=(h, w), **kargs_up)
                tdispL1_norm = tdispL1*factor

                imL_wrap1 = utils.imwrap(imRs1[idx_level], tdispL1_norm, rect1)
                
                tloss1, similary1 = self._loss_ASnC1(imLs1[idx_level], imL_wrap1, tdispL1)
                tsimilary = similary1
                loss = loss + tloss1*self.weights[idx_max-idx]

                # imshow
                if(visualize_disp):
                    tensors = [imLs1[idx_level], imL_wrap1, tdispL1, ]
                    plt = utils.plot_tensors(*tensors)
                    plt.title('scale=%d, tsimilar=%.2f, threshold=%.2f' % (level, tsimilary, threshold))
                    plt.show()
        
                if self.flag_FCTF and (tsimilary < threshold): 
                    flag_break = True; break
            
            if(flag_break): break
        
        return loss
    

    # losses for LUSV-ASn-EC
    def _losses_ASnEC(self, scales, imR1, imL1, dispLs1, edges1, cons1, rect1):
        
        ndim = imL1.dim()

        uplevel = 3
        levels = max(scales) - uplevel + 1
        imLs1 = self._img_pyramid(imL1, levels)
        imRs1 = self._img_pyramid(imR1, levels)
        
        # compute loss
        loss = 0 
        threshold = 0.5
        kargs_up = {'mode': 'bilinear', 'align_corners': True}

        for level, tdispLs1, tedges1, tcons1 in zip(scales, dispLs1, edges1, cons1):
            
            # accumlate loss on scale of level
            flag_break = False
            if(not isinstance(tdispLs1, list)):
                if(tdispLs1.dim() != ndim):
                    continue
                tdispLs1 = [tdispLs1]
                tedges1 = [tedges1]
                tcons1 = [tcons1]
            idx_max = len(tdispLs1)-1
            
            idx_level = max(level-uplevel, 0)
            factor = 2.0/(imRs1[idx_level].size(-1) - 1)
            for idx, (tdispL1, tedge1, tcon1) in enumerate(zip(tdispLs1, tedges1, tcons1)):

                if(level > 0):
                    factor_up = 2**(level - idx_level)
                    h, w = imLs1[idx_level].shape[-2:]
                    tdispL1 = F.interpolate(tdispL1*factor_up, size=(h, w), **kargs_up)
                    tedge1 = tedge1 if(4 > tedge1.dim()) else F.interpolate(tedge1, size=(h, w), **kargs_up)
                    tcon1 = tcon1 if(4 > tcon1.dim()) else F.interpolate(tcon1, size=(h, w), **kargs_up)
                tdispL1_norm = tdispL1*factor

                imL_wrap1 = utils.imwrap(imRs1[idx_level], tdispL1_norm, rect1)
                
                tloss1, similary1 = self._loss_edge_con(imLs1[idx_level], imL_wrap1, tdispL1, tedge1, tcon1)
                tsimilary = similary1
                loss = loss + tloss1*self.weights[idx_max-idx]

                # imshow
                if(visualize_disp):
                    tensors = [imLs1[idx_level], imL_wrap1, tdispL1, ]
                    if(2<tedge1.dim()):
                        tensors += [tedge1.sum(dim=1, keepdim=True), tcon1]
                    plt = utils.plot_tensors(*tensors)
                    plt.title('scale=%d, tsimilar=%.2f, threshold=%.2f' % (level, tsimilary, threshold))
                    plt.show()
        
                if self.flag_FCTF and (tsimilary < threshold): 
                    flag_break = True; break
            
            if(flag_break): break
        
        return loss
    

    # losses for all
    def forward(self, args):

        flag_spported = True
        if(self.flag_supervised):
            
            if('sl1' == self.tname):
                return self._losses_SL1(**args)
            elif('ce' == self.tname):
                return self._losses_CE(**args)
            elif('sl1+ce' == self.tname):
                return self._losses_CE_SL1(**args)
            else:
                flag_spported = False

        elif(self.flag_unsupervised_L):


            if(self.flag_ec):
                return self._losses_ASnEC(**args)
            elif('a' in self.tname):
                return self._losses_ASn(**args)
            else:
                flag_spported = False

        elif(self.flag_unsupervised_LR):

            if('c1' in self.tname):
                return self._losses_ASnC1(**args)
            elif('c2' in self.tname):
                return self._losses_ASnC2(**args)
            else:
                flag_spported = False

        else:
            flag_spported = False

        assert flag_spported, 'unspported lossfun: %s' % self.name


