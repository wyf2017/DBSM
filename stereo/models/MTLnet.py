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


visualize_attention = False # True # 
visualize_refine = False # True # 
visualize_disps = False # True # 

ActiveFun = nn.LeakyReLU(negative_slope=0.1, inplace=True) # nn.ReLU(inplace=True) # 
NormFun2d = nn.BatchNorm2d # nn.InstanceNorm2d # 
NormFun3d = nn.BatchNorm3d # nn.InstanceNorm3d # 
count=0


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


class SimpleResidual2d(nn.Module):
    """
    2D SimpleResidual
    >>> module = SimpleResidual2d(1, dilation=2)
    >>> x = torch.rand(1, 1, 5, 5)
    >>> list(module(x).shape)
    [1, 1, 5, 5]
    """

    def __init__(self, planes, dilation=1):
        super(SimpleResidual2d, self).__init__()
        
        self.planes = planes
        self.conv1 = conv3x3(planes, planes, dilation=dilation, bias=False)
        self.bn1 = NormFun2d(planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation, bias=False)
        self.bn2 = NormFun2d(planes)
        self.ActiveFun = ActiveFun


    def forward(self, x):

        out = self.bn1(self.conv1(x))
        out = self.ActiveFun(out)
        out = self.bn2(self.conv2(out))
        out += x
        out = self.ActiveFun(out)

        return out


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


class AttentionSPP(nn.Module):
    """Spatial pyramid pooling
    >>> feature = torch.rand(2, 16, 5, 5)
    >>> msf = AttentionSPP(16, nLevel=4, kernel_size_first=2)
    >>> out = msf(feature)
    >>> list(out.shape)
    [2, 16, 5, 5]
    """

    def __init__(self, planes, nLevel=4, kernel_size_first=2, attention=True):
        super(AttentionSPP, self).__init__()

        self.planes = planes
        self.nLevel = nLevel
        self.kernel_size_first = kernel_size_first
        self.attention = attention

        ks = [kernel_size_first*(2**i) for i in range(nLevel)]
        self.avg_pools = nn.ModuleList([nn.AvgPool2d(tks) for tks in ks]) # , ceil_mode=True
        self.branchs = nn.ModuleList([conv3x3_bn(planes, planes) for i in range(nLevel + 1)])

        self.lastconv = conv3x3_bn(planes*(nLevel+1), planes)


    def forward(self, x):

        
        h, w = x.shape[-2:]
        kargs_up = {'size':(h, w), 'mode':'bilinear', 'align_corners':True}
        upsamples = [lambda x: x] + [lambda x: F.interpolate(x, **kargs_up)]*self.nLevel
        avg_pools = [lambda x: x] + list(self.avg_pools)
        fun_branch = lambda upsample, branch, avg_pool: upsample(branch(avg_pool(x)))
        branchs = list(map(fun_branch, upsamples, self.branchs, avg_pools))

        if(not self.attention):
            return self.lastconv(torch.cat(branchs, dim=1))
            
        weights = torch.cat([branchs[i][:, :1] for i in range(len(branchs))], dim=1)
        weights = torch.softmax(weights, dim=1)
        branchs = [branchs[i][:, 1:]*weights[:, i:i+1] for i in range(len(branchs))]
        output = self.lastconv(torch.cat([weights] + branchs, dim=1))

        # visualize attention
        if(visualize_attention):
            imgs = weights[:1]
            h, w = imgs.shape[-2:]
            imgs = imgs.view(-1, 1, h, w)
            pad = max(1, max(h, w)//100)
            imgs = make_grid(imgs, nrow=1, padding=pad, normalize=False)
            timg = imgs.data.cpu().numpy()[0]
            #path_save = 'z_ASPP_%d_%d.png' % (h//10, w//10)
            #plt.imsave(path_save, timg)
            plt.subplot(111); plt.imshow(timg)
            plt.show()

        return output


class MultiTaskModule(nn.Module):
    def __init__(self, in_planes, planes=32, branch=1, flag_multi=True, flag_simple=False):
        super(MultiTaskModule, self).__init__()
        
        self.in_planes = in_planes
        self.planes = planes
        self.branch = max(1, branch)
        self.flag_multi = flag_multi
        self.flag_simple = flag_simple
        
        conv_first = None

        if(self.flag_multi):

            self.conv_edgeL = BM.SequentialEx(
                    conv3x3_bn(self.in_planes + 1, self.planes), 
                    conv3x3_bn(self.planes, self.planes), 
                    )
            self.out_edgeL = BM.SequentialEx(
                    conv3x3(self.planes, 1, bias=False), 
                    nn.Sigmoid(), 
                    )

            self.conv_edge_con = BM.SequentialEx(
                    conv3x3_bn(self.in_planes*2 + self.planes + 2, self.planes), 
                    conv3x3_bn(self.planes, self.planes), 
                    )

            self.out_edge_con = BM.SequentialEx(
                    conv3x3_bn(self.planes, self.planes*2, groups=2), 
                    conv3x3(self.planes*2, 2, bias=False, groups=2),
                    nn.Sigmoid(), 
                    )
            
            conv_first = conv3x3_bn(self.in_planes*2 + self.planes*2 + 1, self.planes)

        else:
            
            conv_first = conv3x3_bn(self.in_planes*2 + 1, self.planes)

        if(self.flag_simple):
            
            self.conv_d = BM.SequentialEx(
                    conv_first, 
                    conv3x3_bn(self.planes, self.planes), 
                    conv3x3(planes, 1, bias=False),
                    )

        else:

            self.planes_b = planes*(self.branch)
            self.planes_o = self.branch*2 if self.branch>1 else 1

            self.conv_d = BM.SequentialEx(
                    conv_first, 
                    conv3x3_bn(self.planes, self.planes, dilation=2), 
                    conv3x3_bn(self.planes, self.planes, dilation=4), 
                    conv3x3_bn(self.planes, self.planes, dilation=8), 
                    conv3x3_bn(self.planes, self.planes, dilation=16), 
                    conv3x3_bn(self.planes, self.planes_b), 
                    conv3x3_bn(self.planes_b, self.planes_b, groups=self.branch), 
                    conv3x3(self.planes_b, self.planes_o, groups=self.branch, bias=False),
                    )


    def forward(self, fL, fR, disp, Iedge, Dedge_con):

        disp = disp.detach()
        factor = 2.0/(disp.size(-1) - 1)
        fL_wrap = BM.imwrap(fR.detach(), disp*factor)
        fx = torch.cat([fL, fL_wrap], dim=1)
        
        if(self.flag_multi):

            fe = self.conv_edgeL(torch.cat([fL, Iedge], dim=1))
            Iedge = self.out_edgeL(fe)
            fec = self.conv_edge_con(torch.cat([fx, fe, Dedge_con], dim=1))
            Dedge_con = self.out_edge_con(fec)
            fx = torch.cat([fx, fe, fec, disp], dim=1)

        else:

            fx = torch.cat([fx, disp], dim=1)

        D_res = self.conv_d(fx)

        if(not self.flag_simple) and (1 < self.branch):

            D_b = D_res[:, ::2]
            weight_b = D_res[:, 1::2]
            weight_b = F.softmax(weight_b, dim=1)
            D_res = (weight_b*D_b).sum(dim=1, keepdim=True)
            
        # visualize weight of branch
        if(visualize_refine):

            datas = []
            if(not self.flag_simple) and (1 < self.branch):
                datas.extend([weight_b[:1], D_b[:1], weight_b[:1]*D_b[:1], ])
            datas.append(D_res[:1])
            
            if(self.flag_multi): 
                datas.append(torch.cat([Iedge[:1, ], Dedge_con[:1, 0:1], Dedge_con[:1, 1:2], ], dim=-1))

            if(datas[0].size(1) > 0):
                h, w = datas[0].shape[-2:]
                pad = max(1, max(h, w)//100)
                mw = -datas[0].view(-1, h*w).mean(dim=1)
                _, idxs = torch.sort(mw, descending=False)
                for i in range(3):
                    datas[i] = datas[i][idxs]
            plt.subplot(111)

            for idx, imgs in enumerate(datas):

                shape_view = [-1, 1] + list(imgs.shape[-2:])
                imgs = imgs.view(shape_view).detach().clamp(-6, 6)
                imgs = make_grid(imgs, nrow=D_b.size(1), padding=pad, normalize=False)
                timg = imgs[0].data.cpu().numpy()
                #path_save = 'z_branch_%d_%d_%d.png' % (h//10, w//10, idx)
                #plt.imsave(path_save, timg)
                plt.subplot(len(datas), 1, idx+1); plt.imshow(timg)

            plt.show()


        return D_res, Iedge, Dedge_con


#--------zLAPnet and two variant(zLAPnetF, zLAPnetR)---------#
class MTLnet(BM.BaseModule):
    '''Stereo Matching with Multi-Task Learning'''

    def __init__(self, args, str_kargs='S5B5FMT-ASPP'):
        super(MTLnet, self).__init__()
        
        assert self._lossfun_is_valid(args.loss_name), \
            'invalid lossfun [ model: %s, lossfun: %s]'%(args.arch, args.loss_name)
        self.flag_supervised= args.loss_name.lower().startswith('sv')
        self.flag_ec = ('-ec' in args.loss_name.lower())
        
        self.maxdisp = args.maxdisp
        self.flag_FC = args.flag_FC

        kargs = self._parse_kargs(str_kargs)
        self.nScale = min(7, max(2, kargs[0]))
        self.nBranch = kargs[1]
        self.flag_fusion = kargs[2]
        self.flag_multi = kargs[3]
        self.mode_spp = kargs[4]
        self.scales = [7, 6, 5, 4, 3, 2, 1, 0, ][-(self.nScale+1):]
        
        k = 2**(self.nScale)
        self.shift = int(self.maxdisp//k) + 1
        self.disp_step = float(k)
        
        self.modules_weight_decay = []
        self.modules_conv = []
        
        # feature extration for cost
        self.iters = [1, 2, 2, 2, 2, 2, 2][:self.nScale]
        fn1s = [3, 32, 32, 32, 32, 32, 32][:self.nScale]
        fn2s = (fn1s[1:] + fn1s[-1:])
        SPPs = ([False, True, False, False, False, False, False])[:self.nScale]
        self.convs = nn.ModuleList(map(self._conv_down2_SPP, fn1s, fn2s, SPPs))
        self.modules_conv += [self.convs]

        # feature fuse for refine
        fn1s_r = [n1 + n2 for n1, n2 in zip(fn1s, fn2s)] if self.flag_fusion else fn1s
        fn2s_r = [16] + fn1s[1:]
        self.convs_r = nn.ModuleList(map(conv3x3_bn, fn1s_r, fn2s_r))
        self.modules_conv += [self.convs_r]
        
        # cost_compute for intializing disparity
        self.cost_compute = self._estimator(fn1s[-1]*2, fn1s[-1])
        self.modules_conv += [self.cost_compute ]

        # refines
        fn1s_rf = fn2s_r
        fn2s_rf = fn2s_r
        branchs = [self.nBranch]*len(fn1s_rf)
        #flag_multis = [False] + [self.flag_multi]*(len(branchs) - 1)
        flag_multis = [self.flag_multi]*len(branchs)
        flag_simples = [True] + [False]*(len(branchs) - 1)
        refines_b = list(map(MultiTaskModule, fn1s_rf, fn2s_rf, branchs, flag_multis, flag_simples))
        self.refines = nn.ModuleList(refines_b)
        self.modules_conv += [self.refines]
        
        # init weight
        self.modules_init_()
        

    def _lossfun_is_valid(self, loss_name):

        loss_name = loss_name.lower()
        invalid = (loss_name.startswith('sv')) and ('ce' in loss_name)
        return (not invalid)


    @property
    def name(self):
        tname = '%s_S%dB%d' % (self._get_name(), self.nScale, self.nBranch)
        if(self.flag_fusion):
            tname += 'F'
        if(self.flag_multi):
            tname += 'MT'
        if not ('none'==self.mode_spp):
            tname += '-'+self.mode_spp.upper()
        return tname


    def _parse_kargs(self, str_kargs=None):

        if(str_kargs is None):
            nScale, nBranch, flag_fusion, flag_multi, mode_spp = 5, 5, True, True, 'aspp'
        
        else:
            str_kargs = str_kargs.lower()
            nScale, nBranch = 5, 1
            mode_spp = 'none'

            regex_args = re.compile(r's(\d+)b(\d+)')
            res = regex_args.findall(str_kargs)
            assert 1 == len(res), str(res)
            nScale, nBranch = int(res[0][0]), int(res[0][1])

            flag_fusion = ('f' in str_kargs)
            flag_multi = ('mt' in str_kargs)

            if('aspp' in str_kargs):
                mode_spp = 'aspp'
            elif('spp' in str_kargs):
                mode_spp = 'spp'
        
        msg = 'kargs of model[%s] as follow: \n' % self._get_name()
        msg += ' nScale, nBranch, flag_fusion, flag_multi, mode_spp \n'
        msg += str([nScale, nBranch, flag_fusion, flag_multi, mode_spp]) + '\n'
        logger.info(msg)

        return nScale, nBranch, flag_fusion, flag_multi, mode_spp


    def _conv_down2_SPP(self, in_planes, planes, spp=False):

        layers = [conv3x3_bn(in_planes, planes, stride=2) ]
        if(not spp): 
            layers.append(SimpleResidual2d(planes))
            return BM.SequentialEx(*layers)
        
        mode = self.mode_spp.lower()
        assert mode in ['none', 'spp', 'aspp', 'gaspp']
        if 'none'==mode:
            layers.append(SimpleResidual2d(planes))

        elif 'aspp'==mode:
            layers.append(AttentionSPP(planes, nLevel=4, kernel_size_first=4, attention=True))
        
        elif 'spp'==mode:
            layers.append(AttentionSPP(planes, nLevel=4, kernel_size_first=4, attention=False))

        return BM.SequentialEx(*layers)


    def _estimator(self, in_planes, planes):

        return BM.SequentialEx(
                conv3x3x3_bn(in_planes, planes), 
                SimpleResidual3d(planes), 
                conv3x3x3(planes, 1, bias=False), 
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

        Iedge, Dedge_con = invalid_value, invalid_value
        disps = [disp]
        edges = [Dedge_con]
        cons = [Dedge_con]

        for i in range(self.nScale-1, -1, -1):
            
            # feature fusion---inverse
            if(self.flag_fusion):
                x = BM.upsample_as_bilinear(x, convs[i])
                x = self.convs_r[i](torch.cat([convs[i], x], dim=1))
            else:
                x = self.convs_r[i](convs[i])
            convs[i] = x

            disp = BM.upsample_as_bilinear(disp.detach().clamp(0)*2.0, x)
            if self.flag_multi:
                if(4 > Iedge.dim()):
                    Iedge = torch.zeros_like(disp)
                    Dedge_con = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
                else:
                    Iedge = BM.upsample_as_bilinear(Iedge.detach(), x)
                    Dedge_con = BM.upsample_as_bilinear(Dedge_con.detach(), x)
            mRefine = self.refines[i]

            #iter = 1 if self.training else self.iters[i]
            iter = self.iters[i]
            disps_iter = []
            edges_iter = []
            cons_iter = []

            for ti in range(iter):

                # refine with wraped feature
                disp_r, Iedge, Dedge_con = mRefine(x[:bn], x[bn:], disp, Iedge, Dedge_con)
                disp = disp + disp_r

                disps_iter.append(disp)
                edges_iter.append(0.5*(Iedge+Dedge_con[:, 0:1]))
                cons_iter.append(Dedge_con[:, 1:2])

            disps.insert(0, disps_iter)
            edges.insert(0, edges_iter)
            cons.insert(0, cons_iter)


        # visualize_disps
        if(visualize_disps):

            plt.subplot(111)
            to_numpy = lambda tensor: tensor.cpu().data.numpy()
            col = 1
            row = 1 + (len(disps)+col-1)//col

            plt.subplot(row, col, 1); plt.imshow(to_numpy(BM.normalize(imL[0])).transpose(1, 2, 0))
            plt.subplot(row, col, 2); plt.imshow(to_numpy(BM.normalize(imR[0])).transpose(1, 2, 0))

            for idx, [tdisp, tcon, tedge] in enumerate(zip(disps, cons, edges)):

                if isinstance(tdisp, list):
                    tdisp = torch.cat(tdisp, dim=-1)
                    tcon = torch.cat(tcon, dim=-1) if(self.flag_multi) else invalid_value
                    tedge = torch.cat(tedge, dim=-1) if(self.flag_multi) else invalid_value

                timg = to_numpy(tdisp[0, 0])
                #plt.imsave('z_disp%d.png'%idx, timg)
                plt.subplot(row, 4*col, 4*idx+1); plt.imshow(timg);

                if(3 < tcon.dim()):
                    timg = to_numpy(tcon[0, 0])
                    plt.subplot(row, 3*col, 3*idx+2); plt.imshow(timg);
                    timg = to_numpy(tedge[0, 0])
                    plt.subplot(row, 3*col, 3*idx+3); plt.imshow(timg);

            plt.show()
        
        # return
        if(self.training):

            loss_ex = invalid_value

            if(self.flag_multi) and (not self.flag_ec):

                for tx, tdisps, tcons, tedges in zip(convs, disps, cons, edges):
                    tx, factor = tx.detach(), 2.0/(tx.size(-1) - 1)
                    if(not isinstance(tdisps, list)):
                        tdisps, tcons, tedges = [tdisps], [tcons], [tedges]
                    for tdisp, tcon, tedge in zip(tdisps, tcons, tedges):
                        tloss = self._loss_feature(tdisp*factor, tx[:bn], tx[bn:], tcon)
                        if(tloss > 0): loss_ex += tloss
                        tloss = self._loss_disp_smooth(tdisp, tedge)
                        if(tloss > 0): loss_ex += tloss*0.1

            elif(self.flag_FC and self.flag_supervised):

                for tx, tdisps in zip(convs[1:], disps[1:]):
                    tx, factor = tx.detach(), 2.0/(x.size(-1) - 1)
                    if(not isinstance(tdisps, list)):
                        tdisps = [tdisps]
                    for tdisp in tdisps:
                        tloss = self._loss_feature(tdisp*factor, tx[:bn], tx[bn:])
                        if(tloss > 0): loss_ex += tloss*0.1

            if(self.flag_ec):
                disps.reverse()
                edges.reverse()
                cons.reverse()
                return loss_ex, disps, edges, cons
            else:
                disps.reverse()
                return loss_ex, disps

        else:

            return disps[0][0].clamp(0)



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
    list_name = [
                'MTLnet_S5B5FMT-ASPP', 'MTLnet_S5B5F-ASPP', 'MTLnet_S5B5MT-ASPP', 
                'MTLnet_S5B5FMT-SPP', 'MTLnet_S5B5FMT', 
                ]

    for name in list_name:
        args.arch = name
        model = get_model_by_name(args)
        logger.info('%s passed!\n ' % model.name)

