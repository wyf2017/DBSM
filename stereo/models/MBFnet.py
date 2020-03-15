#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import BaseModule as BM

import traceback
import logging
logger = logging.getLogger(__name__)


visualize_refine = False # True # 
visualize_disps = False # True # 

ActiveFun = nn.LeakyReLU(negative_slope=0.1, inplace=True) # nn.ReLU(inplace=True) # 
NormFun2d = nn.BatchNorm2d # nn.InstanceNorm2d # 
NormFun3d = nn.BatchNorm3d # nn.InstanceNorm3d # 


def conv3x3(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding
    >>> module = conv3x3(1, 1, stride=2, groups=1, bias=True) # dilation=1, 
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('dilation', 1)
    kargs['padding'] = kargs['dilation']
    return nn.Conv2d(in_channel, out_channel, 3, **kargs)


def conv3x3_bn(in_channel, out_channel, **kargs):
    """
    3x3 Conv2d with padding, BatchNorm and ActiveFun
    >>> module = conv3x3_bn(1, 1, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
                conv3x3(in_channel, out_channel, **kargs), 
                NormFun2d(out_channel), ActiveFun, )


def conv3x3x3(in_channel, out_channel, **kargs):
    """
    3x3x3 Conv3d with padding
    >>> module = conv3x3x3(1, 1, stride=2, groups=1, bias=True) # dilation=1, 
    >>> x = torch.rand(1, 1, 5, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3, 3]
    """
    
    kargs.setdefault('dilation', 1)
    kargs['padding'] = kargs['dilation']
    return nn.Conv3d(in_channel, out_channel, 3, **kargs)


def conv3x3x3_bn(in_channel, out_channel, **kargs):
    """
    3x3x3 Conv3d with padding, BatchNorm and ActiveFun
    >>> module = conv3x3x3_bn(1, 1, stride=2, dilation=1, groups=1)
    >>> x = torch.rand(1, 1, 5, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 3, 3, 3]
    """
    kargs.setdefault('bias', False)
    return nn.Sequential(
                conv3x3x3(in_channel, out_channel, **kargs), 
                NormFun3d(out_channel), ActiveFun, )


class SimpleResidual3d(nn.Module):
    """
    3D SimpleResidual
    >>> module = SimpleResidual3d(1, dilation=2)
    >>> x = torch.rand(1, 1, 5, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 5, 5, 5]
    """

    def __init__(self, planes, dilation=1):
        super(SimpleResidual3d, self).__init__()
        
        self.planes = planes
        self.conv1 = conv3x3x3(planes, planes, dilation=dilation, bias=False)
        self.bn1 = NormFun3d(planes)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation, bias=False)
        self.bn2 = NormFun3d(planes)
        self.ActiveFun = ActiveFun


    def forward(self, x):

        out = self.bn1(self.conv1(x))
        out = self.ActiveFun(out)
        out = self.bn2(self.conv2(out))
        out += x
        out = self.ActiveFun(out)

        return out


class SpatialPyramidPooling(nn.Module):
    """Spatial pyramid pooling
    >>> feature = torch.rand(2, 16, 5, 5)
    >>> msf = SpatialPyramidPooling(16, 4)
    >>> out = msf(feature)
    >>> list(out.shape)
    [2, 16, 5, 5]
    """

    def __init__(self, planes, kernel_size_first=4):
        super(SpatialPyramidPooling, self).__init__()

        self.planes = planes
        
        ks = [kernel_size_first*(2**i) for i in range(4)]
        self.avg_pools = nn.ModuleList([nn.AvgPool2d(tks) for tks in ks]) # , ceil_mode=True
        self.branchs = nn.ModuleList([conv3x3_bn(planes, planes) for i in range(4)])
        self.lastconv = conv3x3_bn(planes*5, planes)


    def forward(self, x):
        
        h, w = x.shape[-2:]
        upfun = lambda feat: F.interpolate(feat, (h, w), mode='bilinear', align_corners=True)
        fun_branch = lambda branch, avg_pool: upfun(branch(avg_pool(x)))
        branchs = [x] + list(map(fun_branch, self.branchs, self.avg_pools))
        output = self.lastconv(torch.cat(branchs, 1))

        return output


class MultiBranchRefine(nn.Module):
    def __init__(self, in_planes, planes=32, branch=1):
        super(MultiBranchRefine, self).__init__()
        
        self.branch = max(1, branch)
        self.in_planes = in_planes
        self.planes = planes
        self.planes_b = planes*self.branch
        self.planes_o = self.branch*2 if self.branch>1 else 1
        
        self.conv = BM.SequentialEx(
                conv3x3_bn(self.in_planes, self.planes, dilation=1), 
                conv3x3_bn(self.planes, self.planes, dilation=2), 
                conv3x3_bn(self.planes, self.planes, dilation=4), 
                conv3x3_bn(self.planes, self.planes, dilation=8), 
                conv3x3_bn(self.planes, self.planes, dilation=16), 
                conv3x3_bn(self.planes, self.planes_b, dilation=1), 
                conv3x3_bn(self.planes_b, self.planes_b, groups=self.branch), 
                conv3x3(self.planes_b, self.planes_o, groups=self.branch, bias=False),
                )


    def forward(self, input):

        out = self.conv(input)
        if(1 < self.branch):
            out_b = out[:, ::2]
            weight_b = out[:, 1::2]
            weight_b = F.softmax(weight_b, dim=1)
            out = (weight_b*out_b).sum(dim=1, keepdim=True)
            
            # visualize weight of branch
            if(visualize_refine):
                datas = [weight_b[:1], out_b[:1], weight_b[:1]*out_b[:1], out[:1], ]
                h, w = datas[0].shape[-2:]
                pad = max(1, max(h, w)//100)
                mw = -datas[0].view(-1, h*w).mean(dim=1)
                _, idxs = torch.sort(mw, descending=False)
                plt.subplot(111)
                for idx, imgs in enumerate(datas):
                    shape_view = [-1, 1] + list(imgs.shape[-2:])
                    imgs = imgs.view(shape_view).detach()
                    imgs = imgs.transpose(0, 1).contiguous().view(shape_view).clamp(-6, 6)
                    if(len(idxs)==len(imgs)):
                        imgs = imgs[idxs]
                    imgs = make_grid(imgs, nrow=out_b.size(1), padding=pad, normalize=False)
                    timg = imgs[0].data.cpu().numpy()
                    path_save = 'z_branch_%d_%d_%d.png' % (h//10, w//10, idx)
                    plt.imsave(path_save, timg)
                    plt.subplot(4, 1, idx+1); plt.imshow(timg)
                plt.show()

        else:
            # visualize refine
            if(visualize_refine):
                imgs = out[:1].detach().clamp(-6, 6)
                h, w = imgs.shape[-2:]
                pad = max(1, max(h, w)//100)
                imgs = make_grid(imgs, nrow=1, padding=pad, normalize=False)
                timg = imgs[0].data.cpu().numpy()
                path_save = 'z_refine_%d_%d.png' % (h//10, w//10)
                plt.imsave(path_save, timg)
                plt.subplot(111); plt.imshow(timg)
                plt.show()

        return out


def corration1d_r(fL, fR, radius, stride=1):
    """
    corration of left feature and shift right feature
    
    corration1d(tensor, shift=1, dim=-1) --> tensor of 4D corration

    Args:

        fL: 4D left feature
        fR: 4D right feature
        radius: radius of shift right feature
        stride: stride of shift right feature


    Examples:
    
        >>> x = torch.rand(1, 3, 32, 32)
        >>> y = corration1d_r(x, x, radius=10, stride=1)
        >>> list(y.shape)
        [1, 21, 32, 32]
    """
 
    bn, c, h, w = fL.shape
    shift = radius*2+1
    corrmap = torch.zeros(bn, shift, h, w).type_as(fL.data)
    for i in range(-radius, radius+1):
        idx0 = radius + i
        idx = i*stride
        if(0 == idx):
            corrmap[:, idx0, :, :] = (fL*fR).mean(dim=1)
        elif(idx < 0):
            corrmap[:, idx0, :, :idx] = (fL[..., :idx]*fR[..., -idx:]).mean(dim=1)
        else:
            corrmap[:, idx0, :, idx:] = (fL[..., idx:]*fR[..., :-idx]).mean(dim=1)
    
    return corrmap
    

#--------zLAPnet and two variant(zLAPnetF, zLAPnetR)---------#
class MBFnet(BM.BaseModule):
    '''Stereo Matching based on Multi-branch Fusion'''

    def __init__(self, args, str_kargs='S4B1W'):
        super(MBFnet, self).__init__()
        
        assert self._lossfun_is_valid(args.loss_name), \
            'invalid lossfun [ model: %s, lossfun: %s]'%(args.arch, args.loss_name)
        self.flag_supervised = args.loss_name.lower().startswith('sv')
        
        self.maxdisp = args.maxdisp
        self.flag_FC = args.flag_FC

        kargs = self._parse_kargs(str_kargs)
        self.nScale = min(7, max(2, kargs[0]))
        self.nBranch = kargs[1]
        self.flag_wrap, self.flag_corr = kargs[2]
        self.flag_fusion = kargs[3]
        self.flag_spp = kargs[4]
        self.scales = [7, 6, 5, 4, 3, 2, 1, 0, ][-(self.nScale+1):]
        
        k = 2**(self.nScale)
        self.shift = int(self.maxdisp//k) + 1
        self.disp_step = float(k)
        
        # feature extration for cost
        fn1s = [3, 32, 32, 32, 64, 64, 64][:self.nScale]
        fn2s = (fn1s[1:] + fn1s[-1:])
        SPPs = (([False]*2 + [True]*5) if self.flag_spp else ([False]*7))[:self.nScale]
        fks = ([4]*3 + [2]*4)[:self.nScale]
        self.convs = nn.ModuleList(map(self._conv_down2_SPP, fn1s, fn2s, SPPs, fks))
        self.modules_weight_decay = [self.convs]

        # feature fuse for refine
        fn1s_r = [n1 + n2 for n1, n2 in zip(fn1s, fn2s)] if self.flag_fusion else fn1s
        fn2s_r = [16] + fn1s[1:]
        self.convs_r = nn.ModuleList(map(conv3x3_bn, fn1s_r, fn2s_r))
        self.modules_conv = [self.convs_r]
        
        # cost_compute for intializing disparity
        self.cost_compute = self._estimator(fn1s[-1]*2, fn1s[-1])
        self.modules_conv += [self.cost_compute ]

        # refines
        if(self.flag_wrap):
            fn1s_rf = [n1*2 + 1 for n1 in fn2s_r]
        elif(self.flag_corr):
            fn1s_rf = [n1 + 6 for n1 in fn2s_r]
        else:
            fn1s_rf = [n1 + 1 for n1 in fn2s_r]
        fn1s_rf[0] = fn2s_r[0] + 1
        fn2s_rf = fn2s_r
        branchs = [self.nBranch]*(len(fn1s_rf)-1)

        refines_b = list(map(MultiBranchRefine, fn1s_rf[1:], fn2s_rf[1:], branchs))
        refine0 = self._refine_simple(fn1s_rf[0], fn2s_rf[0])
        self.refines = nn.ModuleList([refine0] + refines_b)
        self.modules_conv += [self.refines]
        
        # init weight
        self.modules_init_()
        

    def _lossfun_is_valid(self, loss_name):

        loss_name = loss_name.lower()
        invalid1 = (loss_name.startswith('sv')) and ('ce' in loss_name)
        invalid2 = (loss_name.startswith('lusv')) and ('ec' in loss_name)
        invalid = invalid1 or invalid2
        return (not invalid)



    @property
    def name(self):
        tname = '%s_S%dB%d' % (self._get_name(), self.nScale, self.nBranch)

        if(self.flag_wrap):
            tname = tname+'W'
        elif(self.flag_corr):
            tname = tname+'C'

        tname = tname+'F' if self.flag_fusion else tname
        tname = tname+'-SPP' if self.flag_spp else tname
        return tname


    def _parse_kargs(self, str_kargs=None):

        if(str_kargs is None):
            nScale, nBranch, (flag_wrap, flag_corr), flag_fusion, flag_spp = 5, 5, (True, False), True, True

        else:
            str_kargs = str_kargs.lower()
            nScale, nBranch = 5, 1

            regex_args = re.compile(r's(\d+)b(\d+)')
            res = regex_args.findall(str_kargs)
            assert 1 == len(res), str(res)

            nScale, nBranch = int(res[0][0]), int(res[0][1])
            flag_wrap = 'w' in str_kargs
            flag_corr = 'c' in str_kargs

            flag_fusion = ('f' in str_kargs)
            flag_spp = ('spp' in str_kargs)

        msg = 'kargs of model[%s] as follow: \n ' % self._get_name()
        msg += 'nScale, nBranch, (flag_wrap, flag_corr), flag_fusion, flag_spp \n '
        msg += str([nScale, nBranch, (flag_wrap, flag_corr), flag_fusion, flag_spp]) + '\n '
        logger.info(msg)

        return nScale, nBranch, (flag_wrap, flag_corr), flag_fusion, flag_spp


    def _conv_down2_SPP(self, in_planes, planes, SPP=False, fks=4):

        if SPP:
            return BM.SequentialEx(
                        conv3x3_bn(in_planes, planes, stride=2), 
                        conv3x3_bn(planes   , planes, stride=1), 
                        SpatialPyramidPooling(planes, fks)
                        )
        else:
            return BM.SequentialEx(
                        conv3x3_bn(in_planes, planes, stride=2), 
                        conv3x3_bn(planes   , planes, stride=1), 
                        )


    def _estimator(self, in_planes, planes):

        return BM.SequentialEx(
                conv3x3x3_bn(in_planes, planes), 
                SimpleResidual3d(planes), 
                SimpleResidual3d(planes), 
                conv3x3x3(planes, 1, bias=False), 
                )


    def _refine_simple(self, in_planes, planes):

        return BM.SequentialEx(
                conv3x3_bn(in_planes, planes, dilation=1), 
                conv3x3_bn(planes, planes, dilation=2), 
                conv3x3(planes, 1, bias=False),
                )


    def get_parameters(self, lr=1e-3,  weight_decay=0):
        ms_weight_decay = self.modules_weight_decay
        ms_conv = self.modules_conv
        return self._get_parameters_group(ms_weight_decay, ms_conv, lr,  weight_decay)



    def forward(self, imL, imR):

        # replace None for returns. 
        # returns contain of None will cause exception with using nn.nn.DataParallel
        invalid_value = torch.zeros(1).type_as(imL) 

        bn = imL.size(0)

        # feature extration---forward
        x = torch.cat([imL, imR], dim=0)
        convs = [x]
        for i in range(self.nScale):
            x = self.convs[i](x)
            convs.append(x)
        
        # cost and disp
        shift = min(x.size(-1), self.shift) # x.size(-1)*3//5 # 
        cost = BM.disp_volume_gen(x[:bn], x[bn:], shift, 1)
        cost = self.cost_compute(cost).squeeze(1)
        disp = BM.disp_regression(cost, 1.0)

        disps = [disp]
        for i in range(self.nScale-1, -1, -1):
            
            # feature fusion---inverse
            if(self.flag_fusion):
                x = BM.upsample_as_bilinear(x, convs[i])
                x = self.convs_r[i](torch.cat([convs[i], x], dim=1))
            else:
                x = self.convs_r[i](convs[i])
            convs[i] = x

            disp = BM.upsample_as_bilinear(disp.detach().clamp(0)*2.0, x)
            mRefine = self.refines[i]

            if(not self.flag_wrap) and (not self.flag_corr) or (0==i):

                # refine without wraped left feature
                input = torch.cat([x[:bn], disp], dim=1)
                disp_r = mRefine(input)

            elif(self.flag_corr):

                # refine with wraped feature
                factor = 2.0/(disp.size(-1) - 1)
                fL_wrap = BM.imwrap(x[bn:].detach(), disp*factor)
                corrmap = corration1d_r(x[bn:], fL_wrap, 2)
                input = torch.cat([x[:bn], corrmap, disp], dim=1)
                disp_r = mRefine(input)

            else:

                # refine with wraped feature
                factor = 2.0/(disp.size(-1) - 1)
                fL_wrap = BM.imwrap(x[bn:].detach(), disp*factor)
                input = torch.cat([x[:bn], fL_wrap, disp], dim=1)
                disp_r = mRefine(input)

            disp = disp + disp_r
            disps.insert(0, disp)

        # visualize_disps
        if(visualize_disps):

            plt.subplot(111)
            to_numpy = lambda tensor: tensor.cpu().data.numpy()
            col = 2
            row = (len(disps)+col+1)//col

            plt.subplot(row, col, 1); plt.imshow(to_numpy(BM.normalize(imL[0])).transpose(1, 2, 0))
            plt.subplot(row, col, 2); plt.imshow(to_numpy(BM.normalize(imR[0])).transpose(1, 2, 0))

            for idx, tdisp in enumerate(disps):
                timg = to_numpy(tdisp[0, 0])
                #plt.imsave('z_disp%d.png'%idx, timg)
                plt.subplot(row, col, idx+3); plt.imshow(timg);
            plt.show()
        
        # return
        if(self.training):

            loss_ex = invalid_value

            if(self.flag_FC and self.flag_supervised):
                for tx, tdisp in zip(convs[1:], disps[1:], ):
                    tx, factor = tx.detach(), 2.0/(tx.size(-1) - 1)
                    loss_ex += 0.1*self._loss_feature(tdisp*factor, tx[:bn], tx[bn:], invalid_value)

            disps.reverse()
            return loss_ex, disps
        else:
            return disps[0].clamp(0)



def get_model_by_name(args):
    
    tmp = args.arch.split('_')
    name_class = tmp[0]
    assert 2>=len(tmp)
    str_kargs = tmp[1] if(2==len(tmp)) else None
    try:
        return eval(name_class)(args, str_kargs)
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
    list_name = ['MBFnet_S5B2WF-SPP', 'MBFnet_S5B2WF', 'MBFnet_S5B2W-SPP', 'MBFnet_S5B2CF-SPP', 'MBFnet_S5B2F-SPP', ]

    for name in list_name:
        args.arch = name
        model = get_model_by_name(args)
        logger.info('%s passed!\n ' % model.name)

