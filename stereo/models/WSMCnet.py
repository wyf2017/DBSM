#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import BaseModule as BM


import traceback
import logging
logger = logging.getLogger(__name__)


visualize_disps = False # True # 

ActiveFun = nn.ReLU(inplace=True) # nn.LeakyReLU(negative_slope=0.1, inplace=True) # 
NormFun2d = nn.BatchNorm2d # nn.InstanceNorm2d # 
NormFun3d = nn.BatchNorm3d # nn.InstanceNorm3d # 


def conv2d3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 Conv2d[no bias] with padding and dilation"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock2d, self).__init__()
        
        self.conv1 = conv2d3x3(inplanes, planes, stride, dilation)
        self.bn1 = NormFun2d(planes)
        self.conv2 = conv2d3x3(planes, planes, 1, dilation)
        self.bn2 = NormFun2d(planes)
        
        self.relu = ActiveFun
        out_planes = planes*BasicBlock2d.expansion
        self.downsample = self.m_downsample(inplanes, out_planes, stride)
        self.stride = stride


    def m_downsample(self, in_planes, out_planes, stride):
        downsample = None
        if stride != 1 or in_planes != out_planes:
           downsample = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                    NormFun2d(out_planes),
                    )
        return downsample


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(x) if(self.downsample) else x

        return out


class ResBlock2d(nn.Module):
    
    def __init__(self, block, blocks, inplanes, planes, stride=1, dilation=1):
        super(ResBlock2d, self).__init__()

        layers = [block(inplanes, planes, stride, dilation)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, 1, dilation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.layers(x)

class FeatureExtraction(nn.Module):
    def __init__(self, planes=32):
        super(FeatureExtraction, self).__init__()
        self.firstconv = nn.Sequential(
                            conv2d3x3(3     , planes, stride=2), NormFun2d(planes), ActiveFun, 
                            conv2d3x3(planes, planes, stride=1), NormFun2d(planes), ActiveFun, 
                            conv2d3x3(planes, planes, stride=1), NormFun2d(planes), ActiveFun, 
                            )

        block = BasicBlock2d
        k = block.expansion
        self.layer1 = ResBlock2d(block, 3 , planes*1  , planes*1, stride=1, dilation=1)
        self.layer2 = ResBlock2d(block, 16, planes*1*k, planes*2, stride=2, dilation=1)
        self.layer3 = ResBlock2d(block, 3 , planes*2*k, planes*4, stride=1, dilation=2)
        self.layer4 = ResBlock2d(block, 3 , planes*4*k, planes*4, stride=1, dilation=2)

        self.kernels_size = [8, 16, 32, 64]
        self.branchs = nn.ModuleList([self.branch_create(planes*4, planes) for i in range(4)])

        self.lastconv = nn.Sequential(
                            conv2d3x3(planes*10, planes*4, stride=1), NormFun2d(planes*4), ActiveFun,
                            conv2d3x3(planes*4 , planes*1, stride=1), 
                            )

    def branch_create(self, inplanes=128, outplanes=32):
        return nn.Sequential(
                                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False), 
                                NormFun2d(outplanes), ActiveFun, 
                                )

    def forward(self, x):
        
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        branchs = [output_raw, output_skip]
        h, w = output_skip.shape[-2:]
        for i in range(4):
            kernel_size = self.kernels_size[i]
            branch = F.avg_pool2d(output_skip, kernel_size, padding=kernel_size//2)
            branch = self.branchs[i](branch)
            branch = F.interpolate(branch, (h, w), mode='bilinear', align_corners=True)
            branchs.append(branch)

        output_feature = torch.cat(branchs, 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


def conv3d3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 Conv3d[no bias] with padding and dilation"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


def deconv3d3x3(in_planes, out_planes, stride=2):
    """3x3 ConvTranspose3d[no bias] with padding and dilation"""
    return nn.ConvTranspose3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, output_padding=1, bias=False)


class Hourglass(nn.Module):
    def __init__(self, planes):
        super(Hourglass, self).__init__()

        self.conv1 = nn.Sequential(conv3d3x3(planes*1, planes*2, stride=2), NormFun3d(planes*2), ActiveFun, )
        self.conv2 = nn.Sequential(conv3d3x3(planes*2, planes*2, stride=1), NormFun3d(planes*2), )
        
        self.conv3 = nn.Sequential(conv3d3x3(planes*2, planes*2, stride=2), NormFun3d(planes*2), ActiveFun, )
        self.conv4 = nn.Sequential(conv3d3x3(planes*2, planes*2, stride=1), NormFun3d(planes*2), ActiveFun, )

        self.conv5 = nn.Sequential(deconv3d3x3(planes*2, planes*2, stride=2), NormFun3d(planes*2), ) # +conv2
        self.conv6 = nn.Sequential(deconv3d3x3(planes*2, planes*1, stride=2), NormFun3d(planes*1), ) # +x

        self.relu = ActiveFun


    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        _, _, d1, h1, w1 = x.shape # 1/4
        _, _, d2, h2, w2 = out.shape # 1/8
        
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None: 
            pre = pre + postsqu
        pre = self.relu(pre)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        post = self.conv5(out)[..., :d2, :h2, :w2] #in:1/16 out:1/8
        if presqu is not None:
            post = post + presqu
        else: 
            post = post + pre
        post = self.relu(post)
        logger.debug('self.conv5: %s\n shape of input and output: %s, %s\n' %(
                        str(self.conv5), str(out.shape), str(post.shape) ) )
        assert out.shape[-1]*2 < w2+2
        

        out  = self.conv6(post)[..., :d1, :h1, :w1]  #in:1/8 out:1/4
        logger.debug('self.conv6: %s\n shape of input and output: %s, %s\n' %(
                        str(self.conv6), str(post.shape), str(out.shape) ) )
        assert post.shape[-1]*2 < w1+2

        return out, pre, post


class CostCompute(nn.Module):
    def __init__(self, inplanes=64, planes=32, C=1):
        super(CostCompute, self).__init__()

        self.fun_active = ActiveFun
        self.dres0 = self.dres0_create(inplanes, planes)
        self.dres1 = self.dres_create(planes)

        self.dres2 = Hourglass(planes)
        self.dres3 = Hourglass(planes)
        self.dres4 = Hourglass(planes)

        self.classif1 = self.classify_create(planes, C)
        self.classif2 = self.classify_create(planes, C)
        self.classif3 = self.classify_create(planes, C)

    def dres0_create(self, inplanes, planes):
        return nn.Sequential(
                    conv3d3x3(inplanes, planes, stride=1), NormFun3d(planes), ActiveFun, 
                    conv3d3x3(planes  , planes, stride=1), NormFun3d(planes), ActiveFun, 
                    )

    def dres_create(self, planes):
        return nn.Sequential(
                    conv3d3x3(planes, planes, stride=1), NormFun3d(planes), ActiveFun, 
                    conv3d3x3(planes, planes, stride=1), NormFun3d(planes), 
                    ) 

    def classify_create(self, planes, C=1):
        return nn.Sequential(
                    conv3d3x3(planes, planes, stride=1), NormFun3d(planes), ActiveFun, 
                    conv3d3x3(planes, C     , stride=1),  
                    ) 

    def forward(self, cost):

        cost = self.dres0(cost)
        cost = self.dres1(cost) + cost

        out1, pre1, post1 = self.dres2(cost, None, None) 
        out1 = out1 + cost

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2 + cost

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3 + cost

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        costs = [cost1, cost2, cost3] if self.training else [cost3]
        bn, c, d, h, w = cost3.shape
        if(c > 1):
            costs = [tcost.transpose(1, 2).reshape(bn, c*d, h, w).contiguous() for tcost in costs]
        else:
            costs = [tcost[:, 0] for tcost in costs]
        
        return costs


class WSMCnet(BM.BaseModule):
    
    def __init__(self, args, str_kargs='S2C3F32B'):
        super(WSMCnet, self).__init__()
        
        assert self._lossfun_is_valid(args.loss_name), \
            'invalid lossfun [ model: %s, lossfun: %s]'%(args.arch, args.loss_name)
        
        self.maxdisp = args.maxdisp
        self.flag_FC = args.flag_FC
        
        kargs = self._parse_kargs(str_kargs)
        self.S = max(1, kargs[0])
        self.C = max(1, kargs[1])
        self.F = kargs[2]
        self.flag_bilinear = kargs[3]
        self.shift = (self.maxdisp//(self.S<<2)) + 1

        if(self.flag_bilinear):
            self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.step_disp = float(self.S<<2)/self.C
        else:
            self.upsample = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
            self.step_disp = float(self.S)/self.C

        self.scales = [0]
        self.step_disps = [self.step_disp]
        
        # mode_output ['disps', 'costs', 'multi']
        self.mode_output = 'disps'
        loss_name = args.loss_name.lower()
        if(loss_name.startswith('usv')):
            self.mode_output = 'disps'
        elif(loss_name.startswith('sv')):
            if ('sl1' in loss_name) and ('ce' in loss_name):
                self.mode_output = 'multi'
            elif('ce' in loss_name):
                self.mode_output = 'costs'
            elif('sl1' in loss_name):
                self.mode_output = 'disps'

        # feature_extraction and cost_compute
        self.feature_extraction = FeatureExtraction(planes=32)
        self.cost_compute = CostCompute(inplanes=64, planes=self.F, C=self.C)


    def _lossfun_is_valid(self, loss_name):

        loss_name = loss_name.lower()
        invalid = (loss_name.startswith('lusv')) and ('ec' in loss_name)
        return (not invalid)


    @property
    def name(self):

        tname = '%s_S%dC%dF%d' % (self._get_name(), self.S, self.C, self.F)
        return tname+'B' if self.flag_bilinear else tname


    def _parse_kargs(self, str_kargs=None):

        if(str_kargs is None):
            S, C, F, flag_bilinear = 2, 3, 32, True
        else:

            str_kargs = str_kargs.lower()
            S, C, F = 1, 1, 32

            regex_args = re.compile(r's(\d+)c(\d+)f(\d+)')
            res = regex_args.findall(str_kargs)
            assert 1 == len(res), str(res)

            S, C, F = int(res[0][0]), int(res[0][1]), int(res[0][2])
            flag_bilinear = 'b' in str_kargs
        
        msg = 'kargs of model[%s] as follow: \n' % self._get_name()
        msg += ' S, C, F, flag_bilinear \n'
        msg += str([S, C, F, flag_bilinear]) + '\n'
        logger.info(msg)
        
        return S, C, F, flag_bilinear


    def get_parameters(self, lr=1e-3,  weight_decay=0):
        ms_weight_decay = [self.feature_extraction, ]
        ms_conv = [self.cost_compute, ]
        return self._get_parameters_group(ms_weight_decay, ms_conv, lr,  weight_decay)


    def forward(self, imL, imR):

        # replace None for returns. 
        # returns contain of None will cause exception with using nn.nn.DataParallel
        invalid_value = torch.zeros(1).type_as(imL) 

        bn = imL.size(0)

        # feature extraction
        x = torch.cat([imL, imR], dim=0)
        x = self.feature_extraction(x)

        # compute matching cost
        cost = BM.disp_volume_gen(x[:bn], x[bn:], self.shift, self.S)
        costs = self.cost_compute(cost)
        
        h, w = imL.shape[-2:]
        h2, w2 = x.shape[-2:]
        assert (w2*4 >= w > (w2-1)*4) and (h2*4 >= h > (h2-1)*4), str([h, w, h2, w2])
        if(self.flag_bilinear):
            costs = [self.upsample(tcost)[..., :h, :w] for tcost in costs]
        else:
            costs = [self.upsample(tcost[:, None])[..., 0, :, :h, :w] for tcost in costs]
        
        # visualize_disps
        if(visualize_disps):

            disps = [BM.disp_regression_nearby(tcost, self.step_disp) for tcost in costs]
            plt.subplot(111)
            to_numpy = lambda tensor: tensor.cpu().data.numpy()
            col = 2
            row = (len(disps)+col+1)//col

            plt.subplot(row, col, 1); plt.imshow(to_numpy(BM.normalize(imL[0])).transpose(1, 2, 0))
            
            disps.inverse()
            for idx, tdisp in enumerate(disps):
                timg = to_numpy(tdisp[0, 0])
                #plt.imsave('z_disp%d.png'%idx, timg)
                plt.subplot(row, col, idx+3); plt.imshow(timg);
            plt.show()

        # return
        logger.debug('training: %s \n' % str(self.training))
        if(self.training):
            
            loss_ex = invalid_value
            if('costs' == self.mode_output):
                return loss_ex, [costs]
            elif('disps' == self.mode_output):
                disps = []
                for tcost in costs:
                    disps.append(BM.disp_regression(tcost, self.step_disp))
                    disps.append(BM.disp_regression_nearby(tcost, self.step_disp))
                return loss_ex, [disps]
            elif('multi' == self.mode_output):
                disps = [BM.disp_regression_nearby(tcost, self.step_disp) for tcost in costs]
                return loss_ex, [disps], [costs]
            else:
                raise Exception('unsupported mode_output of WSMCnet')
        
        else:
            return BM.disp_regression_nearby(costs[-1], self.step_disp)


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
    list_name = ['WSMCnet_S2C3F32', 'WSMCnet_S2C3F32B',]

    for name in list_name:
        args.arch = name
        model = get_model_by_name(args)
        logger.info('%s passed!\n ' % model.name + 
                    'Fun_upsample : %s \n '% str(model.upsample) + 
                    'step_disp : %s \n '% str(model.step_disp))

