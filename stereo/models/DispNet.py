#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
#import torch.nn.functional as F
#from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import BaseModule as BM

import traceback
import logging
logger = logging.getLogger(__name__)


visualize_disps = False # True # 

ActiveFun = nn.LeakyReLU(negative_slope=0.1, inplace=True) # nn.ReLU(inplace=True) # 
NormFun2d = nn.BatchNorm2d # nn.InstanceNorm2d # 

def padConv2d(in_channel, out_channel, kernel_size, **kargs):
    """
    Conv2d with padding
    >>> module = padConv2d(1, 1, 5, stride=2, groups=1, bias=True) # dilation=1, 
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('dilation', 1)
    pad = kernel_size//2
    if(kargs['dilation'] > 1): pad *= kargs['dilation']
    kargs['padding'] = pad
    return nn.Conv2d(in_channel, out_channel, kernel_size, **kargs)


def padConv2d_bn(in_channel, out_channel, kernel_size, **kargs):
    """
    Conv2d with padding, BatchNorm and ActiveFun
    >>> module = padConv2d_bn(1, 1, 5, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
                padConv2d(in_channel, out_channel, kernel_size, **kargs), 
                NormFun2d(out_channel), ActiveFun, )


def deconv2d(in_channel, out_channel, kernel_size, stride=2):

    padding = (kernel_size - 1)//2
    output_padding = stride - (kernel_size - 2*padding)
    return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, bias=False, 
                                padding=padding, output_padding=output_padding), 
            ActiveFun, 
            )


def corration1d(fL, fR, shift, stride=1):
    """
    corration of left feature and shift right feature
    
    corration1d(tensor, shift=1, dim=-1) --> tensor of 4D corration

    Args:

        fL: 4D left feature
        fR: 4D right feature
        shift: count of shift right feature
        stride: stride of shift right feature


    Examples:
    
        >>> x = torch.rand(1, 3, 32, 32)
        >>> y = corration1d(x, x, shift=20, stride=1)
        >>> list(y.shape)
        [1, 20, 32, 32]
    """
    
    bn, c, h, w = fL.shape
    corrmap = torch.zeros(bn, shift, h, w).type_as(fL.data)
    for i in range(0, shift):
        idx = i*stride
        corrmap[:, i, :, idx:] = (fL[..., idx:]*fR[..., :w-idx]).mean(dim=1)
    
    return corrmap
    

class DispNet(BM.BaseModule):

    def __init__(self, args):
        super(DispNet, self).__init__()

        assert self._lossfun_is_valid(args.loss_name), \
            'invalid lossfun [ model: %s, lossfun: %s]'%(args.arch, args.loss_name)
        
        self.maxdisp = args.maxdisp
        self.scales = [6, 5, 4, 3, 2, 1, 0, ]

        # Feature extration
        self.conv3b = padConv2d_bn(256, 256, kernel_size=3, stride=1)
        self.conv4a = padConv2d_bn(256, 512, kernel_size=3, stride=2)
        self.conv4b = padConv2d_bn(512, 512, kernel_size=3, stride=1)
        self.conv5a = padConv2d_bn(512, 512, kernel_size=3, stride=2)
        self.conv5b = padConv2d_bn(512, 512, kernel_size=3, stride=1)
        self.conv6a = padConv2d_bn(512, 1024, kernel_size=3, stride=2)
        self.conv6b = padConv2d_bn(1024, 1024, kernel_size=3, stride=1)
        
        # decode and predict disp
        self.pr6 = padConv2d(1024, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv5 = deconv2d(1024, 512, kernel_size=4, stride=2)
        self.iconv5  = padConv2d_bn(1025, 512, kernel_size=3, stride=1)
        self.pr5     = padConv2d(512, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv4 = deconv2d(512, 256, kernel_size=4, stride=2)
        self.iconv4  = padConv2d_bn(769, 256, kernel_size=3, stride=1)
        self.pr4     = padConv2d(256, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv3 = deconv2d(256, 128, kernel_size=4, stride=2)
        self.iconv3  = padConv2d_bn(385, 128, kernel_size=3, stride=1)
        self.pr3     = padConv2d(128, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv2 = deconv2d(128, 64, kernel_size=4, stride=2)
        self.iconv2  = padConv2d_bn(193, 64, kernel_size=3, stride=1)
        self.pr2     = padConv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv1 = deconv2d(64, 32, kernel_size=4, stride=2)
        self.iconv1  = padConv2d_bn(97, 32, kernel_size=3, stride=1)
        self.pr1     = padConv2d(32, 1, kernel_size=3, stride=1, padding=1)


    def _lossfun_is_valid(self, loss_name):

        loss_name = loss_name.lower()
        invalid1 = (loss_name.startswith('sv')) and ('ce' in loss_name)
        invalid2 = (loss_name.startswith('lusv')) and ('ec' in loss_name)
        invalid = invalid1 or invalid2
        return (not invalid)


    def _pr_zoom_k(self, k=0.1):
        for m in [self.pr6, self.pr5, self.pr4, self.pr3, self.pr2, self.pr1]:
            m.weight.data = m.weight.data*k


    def _modules_weight_decay(self):
        return [
                self.conv3b, self.conv4a, self.conv4b, 
                self.conv5a, self.conv5b, 
                ]


    def _modules_conv(self):
        return [
                self.conv6a, self.conv6b, self.pr6, 
                self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5,
                self.iconv1, self.iconv2, self.iconv3, self.iconv4, self.iconv5,
                self.pr1, self.pr2, self.pr3, self.pr4, self.pr5, 
                ]


    def _disp_estimator(self, imL, fconv1, fconv2, fconv3):

        # replace None for returns. 
        # returns contain of None will cause exception with using nn.nn.DataParallel
        invalid_value = torch.zeros(1).type_as(imL) 

        # feature extration
        conv3b = self.conv3b(fconv3)
        conv4a = self.conv4a(conv3b)
        conv4b = self.conv4b(conv4a)
        conv5a = self.conv5a(conv4b)
        conv5b = self.conv5b(conv5a)
        conv6a = self.conv6a(conv5b)
        conv6b = self.conv6b(conv6a)
        
        #　disparity estimator
        disp6 = self.pr6(conv6b)
        
        # disp5
        h, w = conv5b.shape[-2:]
        deconv5 = self.deconv5(conv6b)[..., :h, :w]
        up_disp6 = BM.upsample_as_bilinear(disp6.detach(), conv5b)
        iconv5 = self.iconv5(torch.cat([deconv5, conv5b, up_disp6], 1))
        disp5 = self.pr5(iconv5)

        # disp4
        h, w = conv4b.shape[-2:]
        deconv4 = self.deconv4(iconv5)[..., :h, :w]
        up_disp5 = BM.upsample_as_bilinear(disp5.detach(), conv4b)
        iconv4 = self.iconv4(torch.cat([deconv4, conv4b, up_disp5], 1))
        disp4 = self.pr4(iconv4)

        # disp3
        h, w = conv3b.shape[-2:]
        deconv3 = self.deconv3(iconv4)[..., :h, :w]
        up_disp4 = BM.upsample_as_bilinear(disp4.detach(), conv3b)
        iconv3 = self.iconv3(torch.cat([deconv3, conv3b, up_disp4], 1))
        disp3 = self.pr3(iconv3)

        # disp2
        h, w = fconv2.shape[-2:]
        deconv2 = self.deconv2(iconv3)[..., :h, :w]
        up_disp3 = BM.upsample_as_bilinear(disp3.detach(), fconv2)
        iconv2 = self.iconv2(torch.cat([deconv2, fconv2, up_disp3], 1))
        disp2 = self.pr2(iconv2)

        # disp1
        h, w = fconv1.shape[-2:]
        deconv1 = self.deconv1(iconv2)[..., :h, :w]
        up_disp2 = BM.upsample_as_bilinear(disp2.detach(), fconv1)
        iconv1 = self.iconv1(torch.cat([deconv1, fconv1, up_disp2], 1))
        disp1 = self.pr1(iconv1)

        # disp0
        disp0 = BM.upsample_as_bilinear(disp1, imL)

        # visualize_disps
        if(visualize_disps):

            disps = [disp6, disp5, disp4, disp3, disp2, disp1, disp0]
            plt.subplot(111)
            to_numpy = lambda tensor: tensor.cpu().data.numpy()
            col = 2
            row = (len(disps)+col+1)//col

            plt.subplot(row, col, 1); plt.imshow(to_numpy(BM.normalize(imL[0])).transpose(1, 2, 0))
            
            disps.reverse()
            for idx, tdisp in enumerate(disps):
                timg = to_numpy(tdisp[0, 0])
                plt.imsave('z_disp%d.png'%idx, timg)
                plt.subplot(row, col, idx+3); plt.imshow(timg);
            plt.show()
        
        # return
        logger.debug('training: %s \n' % str(self.training))
        if(self.training):
            loss_ex = invalid_value
            return loss_ex, [disp6/64, disp5/32, disp4/16, disp3/8, disp2/4, invalid_value, disp0]
        else:
            return disp0.clamp(0)


class DispNetS(DispNet):

    def __init__(self, args):
        super(DispNetS, self).__init__(args)

        # 卷积层
        self.conv1 = padConv2d_bn(  6,  64, kernel_size=7, stride=2)
        self.conv2 = padConv2d_bn( 64, 128, kernel_size=5, stride=2)
        self.conv3 = padConv2d_bn(128, 256, kernel_size=5, stride=2)

        # initialize weight
        self.modules_init_()
        self._pr_zoom_k(0.1)

    
    def get_parameters(self, lr=1e-3,  weight_decay=0):

        modules_new = [self.conv1, self.conv2, self.conv3, ]
        modules_weight_decay = modules_new + self._modules_weight_decay()
        modules_conv = self._modules_conv()
        return self._get_parameters_group(modules_weight_decay, modules_conv, lr,  weight_decay)


    def forward(self, imL, imR):

        # feature extration
        x = torch.cat([imL, imR], dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        # disparity estimator
        return self._disp_estimator(imL, conv1, conv2, conv3)


class DispNetC(DispNet):

    def __init__(self, args):
        super(DispNetC, self).__init__(args)

        # maximum of disparity
        self.shift = 40 # 1 + maxdisp//4

        # 卷积层
        self.conv1 = padConv2d_bn(3, 64, kernel_size=7, stride=2)
        self.conv2 = padConv2d_bn(64, 128, kernel_size=5, stride=2)
        self.redir = padConv2d_bn(128, 64, kernel_size=1, stride=1)
        self.conv3 = padConv2d_bn(64 + self.shift, 256, kernel_size=5, stride=2)

        # initialize weight
        self.modules_init_()
        self._pr_zoom_k(0.1)

    def get_parameters(self, lr=1e-3,  weight_decay=0):

        modules_new = [self.conv1, self.conv2, self.redir, self.conv3, ]
        modules_weight_decay = modules_new + self._modules_weight_decay()
        modules_conv = self._modules_conv()
        return self._get_parameters_group(modules_weight_decay, modules_conv, lr,  weight_decay)


    def forward(self, imL, imR):

        # feature extration
        bn = imL.size(0)
        x = torch.cat([imL, imR], dim=0)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        redir = self.redir(conv2[:bn])
        corrmap = corration1d(conv2[:bn], conv2[bn:], self.shift, 1)
        conv3 = self.conv3(torch.cat([redir, corrmap], dim=1))
        
        # disparity estimator
        return self._disp_estimator(imL, conv1[:bn], conv2[:bn], conv3)


def get_model_by_name(args):
    
    tmp = args.arch.split('_')
    name_class = tmp[0]
    try:
        return eval(name_class)(args)
    except:
        raise Exception(traceback.format_exc())


def get_settings():

    import argparse
    parser = argparse.ArgumentParser(description='Deep Stereo Matching by pytorch')

    # arguments of model
    parser.add_argument('--arch', default='DispNetC',
                        help='select arch of model')
    parser.add_argument('--maxdisp', type=int ,default=192,
                        help='maxium disparity')

    # arguments of lossfun
    parser.add_argument('--loss_name', default='SV-SL1',
                        help='name of lossfun [SV-(SL1/CE/SL1+CE), USV-(common/depthmono/SsSMnet/AS1C/AS2C)-mask]')
    parser.add_argument('--flag_FC', action='store_true', default=False,
                        help='enables feature consistency')

    # parser arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':

#    logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

    args = get_settings()
    list_name = ['DispNetS', 'DispNetC' ]

    for name in list_name:
        args.arch = name
        model = get_model_by_name(args)
        logger.info('%s passed!\n ' % model.name)

