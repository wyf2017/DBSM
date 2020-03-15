#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#from torchvision.utils import make_grid


visualize_disp = False # True # 
visualize_wraped = False # True # 
flag_FCTF = not (visualize_wraped or visualize_disp)


class BaseModule(nn.Module):
    
    def __init__(self):
        super(BaseModule, self).__init__()
        
        self.modules_init_ = lambda: modules_init_(self.modules())
        self.get_id_paramertes = get_id_paramertes
        self.parameters_gen = parameters_gen

   
    @property
    def name(self):
        return self._get_name()


    def _lossfun_is_valid(self, loss_name):
        raise NotImplementedError('the function[ _lossfun_is_valid ] of model[%] is not Implemented' % self.name)

    #----------------------the group of parameters for optimer-----------------------------------#
    def _get_parameters_group(self, modules_weight_decay, modules_conv, lr=1e-3,  weight_decay=0):
        
        param_groups = []
        get_parameters = self.parameters_gen
        
        # group_weight_decay
        instance_weight_decay = (nn.Conv1d, nn.Conv2d, nn.Conv3d, )
        params_weight_decay = get_parameters(modules_weight_decay, instance_weight_decay, bias=False)
        group_weight_decay = {'params': params_weight_decay, 'lr': lr*1, 'weight_decay': 1*weight_decay}
        param_groups.append(group_weight_decay)
        
        # group_conv
        instance_conv = (nn.Conv1d, nn.Conv2d, nn.Conv3d,)
        params_conv = get_parameters(modules_conv, instance_conv, bias=False)
        group_conv = {'params': params_conv, 'lr': lr*1, 'weight_decay': 0.1*weight_decay}
        param_groups.append(group_conv)
        
        # group_ConvTranspose
        instance_conv = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,)
        params_conv = get_parameters(modules_conv, instance_conv, bias=False)
        group_conv = {'params': params_conv, 'lr': lr*0.2, 'weight_decay': 0.1*weight_decay}
        param_groups.append(group_conv)
        
        # group_bias
        instance_bias = (nn.Conv1d, nn.ConvTranspose1d, 
                         nn.Conv2d, nn.ConvTranspose2d, 
                         nn.Conv3d, nn.ConvTranspose3d, 
                         nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
                         )
        params_bias = get_parameters([self], instance_bias, bias=True)
        group_conv = {'params': params_bias, 'lr': lr*2, 'weight_decay': 0}
        param_groups.append(group_conv)
        
        # group_bn
        instance_bn = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, )
        params_bn = get_parameters([self], instance_bn, bias=False)
        group_bn = {'params': params_bn, 'lr': lr*1, 'weight_decay': 0}
        param_groups.append(group_bn)
        
        return param_groups
    

    #-------------------compute loss and visualze intermediate result------------------------------#
    def _loss_disp_smooth(self, disp, edge):
        
        
        fun_loss = lambda x: torch.abs(x)
        ds1 = fun_loss(disp[..., 1:-1] - 0.5*(disp[..., :-2] + disp[..., 2:]))
        ds2 = fun_loss(disp[..., 1:-1, :] - 0.5*(disp[..., :-2, :] + disp[..., 2:, :]))
        
        if(3 > edge.dim()):
            
            return 0.5*(ds1.mean() + ds2.mean())

        else:

            bn = edge.size(0)
            mean_edge = edge.view(bn, -1).mean(-1).view([bn, 1, 1, 1]) + 0.01
            loss_ds_edge = lambda ds, edge: (ds*(1-edge)).mean() + (ds*mean_edge).mean()
            tloss1 = loss_ds_edge(ds1, edge[..., 1:-1])
            tloss2 = loss_ds_edge(ds2, edge[..., 1:-1, :])
            
            return 0.5*(tloss1 + tloss2)


    def _loss_feature(self, disp_norm, fL, fR, con):
        
        fL, fR = fL.detach(), fR.detach()
        w_fL = imwrap(fR, disp_norm)

        bn = fL.size(0)
        mfL_dim1 = fL.detach().abs().mean(dim=1, keepdim=True)
        maxfL_dim1, _ = mfL_dim1.view(bn, -1).max(dim=-1)
        maxfL_dim1 = maxfL_dim1.clamp(1)[:, None, None, None]

        dfL = F.l1_loss(w_fL, fL, reduction='none')
        dfL = (dfL.mean(dim=1, keepdim=True)/maxfL_dim1)

        mask = (0 != w_fL.detach().abs().sum(dim=1, keepdim=True))
        if(len(0==mask[mask])): mask = None
        if(3 > con.dim()):

            loss = dfL[mask].mean()
            return loss

        else:

            bn = con.size(0)
            mean_incon = (1 - con.view(bn, -1).mean(-1).view([bn, 1, 1, 1])) + 0.01
            tloss = (dfL*con).mean() + (mean_incon*dfL)[mask].mean()
            return tloss



#----------- SequentialEx -----------------------------------------------------#
def FlattenSequential(*moduls):
    """
    Sequential with Flattened modules
    >>> module = nn.Sequential(nn.Conv2d(1,1,3), nn.Conv2d(1,1,3))
    >>> 2*len(module) == len(FlattenSequential(module, module))
    True
    """
    return nn.Sequential(*SequentialFlatten(*moduls))


def SequentialFlatten(*moduls):
    """
    Flatten modules with Sequential
    >>> module = nn.Sequential(nn.Conv2d(1,1,3), nn.Conv2d(1,1,3))
    >>> 2*len(module) == len(SequentialFlatten(module, module))
    True
    """
    layers = []
    for m in moduls:
        if isinstance(m, nn.Sequential):
            layers.extend(SequentialFlatten(*m))
        elif isinstance(m, nn.Module):
            layers.append(m)
        else:
            msg_err = 'module[ %s ] is not a instance of nn.Module or nn.Sequential' % str(m)
            raise Exception(msg_err)
    return layers


SequentialEx = FlattenSequential

#----------- generator and initialization of parameters for optimer -----------#
def get_id_paramertes(*parameters):
    '''
    get id of parameters

    get_id_paramertes(parameters) --> [ids]

    Args:

        parameters: iterable parameters with nn.Parameter

    Examples:
    
        >>> m1, m2 = nn.Conv2d(1, 10, 3), nn.Conv2d(1, 10, 3)
        >>> m = nn.Sequential(m1, m2)
        >>> ids1 = get_id_paramertes(m1.parameters(), [{'params': m2.parameters()}])
        >>> ids = get_id_paramertes(m.parameters())
        >>> ids == ids1, len(ids1)
        (True, 4)
    '''
    
    ids = []
    for pm in parameters:
        if isinstance(pm, list):
            for tpm in pm: 
                ids.extend(get_id_paramertes(tpm))
        elif isinstance(pm, dict) and pm.get('params'):
            state_dict = torch.optim.Adam([pm], lr=0.001, betas=(0.9, 0.99)).state_dict()
            ids.extend(state_dict['param_groups'][0]['params'])
        else:
            state_dict = torch.optim.Adam(pm, lr=0.001, betas=(0.9, 0.99)).state_dict()
            ids.extend(state_dict['param_groups'][0]['params'])
    ids.sort()
    return ids


def parameters_gen(modules, instance=(nn.Conv2d, ), bias=False):
    '''
    generator of parameters

    parameters_gen(modules, instance=(nn.Conv2d, ), bias=False) --> generator[p]

    Args:

        modules: iterable modules with nn.Module
        instance: type of instance with nn.Parameter

    Examples:
    
        >>> modules = [nn.Conv2d(1, 10, 3) for i in range(5)]
        >>> params = parameters_gen([modules], bias=True)
        >>> len([tp for tp in params]) # len(params) # 
        5
    '''

    if isinstance(modules, nn.Module):
        for m in modules.modules():
            if(isinstance(m, instance)):
                param = m.bias if bias else m.weight
                if (param is not None) and param.requires_grad:
                    yield param
            else:
                pass
    
    elif isinstance(modules, (list, tuple)):
        for m in modules:
            for param in parameters_gen(m, instance, bias):
                yield param


def modules_init_(modules):
    '''
    initialize parameters of modules

    modules_init_(modules) --> None

    Args:

        modules: iterable modules with nn.Module

    Examples:
    
        >>> m = nn.Conv2d(1, 1, 3)
        >>> data1 = m.weight.data.clone()
        >>> modules_init_(m.modules())
        >>> (data1 == m.weight.data).max().item()
        0
        >>> m.bias.data.item()
        0.0
    '''

    for m in modules:
        
        # bias
        if(hasattr(m, 'bias') and m.bias is not None):
            m.bias.data.zero_()
        
        Convs = (nn.Conv3d, nn.Conv2d, nn.Conv1d)
        ConvTransposes = (nn.ConvTranspose3d, nn.ConvTranspose2d, nn.ConvTranspose1d)
        BatchNorms = (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)
        
        if isinstance(m, nn.Linear): # weight of Linear
            v = 1.0/(m.out_features**0.5)
            m.weight.data.uniform_(-v, v)

        elif isinstance(m, Convs): # weight of Conv3d/2d/1d
            weight_init_Conv_(m)
        
        elif isinstance(m, ConvTransposes): # ConvTranspose3d/2d/1d
            weight_init_bilinear_(m)
        
        elif isinstance(m, BatchNorms): # BatchNorm3d/2d/1d
            m.weight.data.fill_(1)


def weight_init_Conv_(m_Conv):
    '''
    initialize weight for nn.Conv

    weight_init_Conv_(m_Conv) --> None

    Args:

        m_Conv: module with type of nn.Conv[1d/2d/3d]

    Examples:
    
        >>> m = nn.Conv2d(1, 1, 3)
        >>> data1 = m.weight.data.clone()
        >>> weight_init_Conv_(m)
        >>> (data1 == m.weight.data).max().item()
        0
    '''

    n = m_Conv.out_channels
    for kz in m_Conv.kernel_size:
        n *= kz
    m_Conv.weight.data.normal_(0, (2.0/n)**0.5)



def weight_init_bilinear_(m_ConvTranspose):
    '''
    make bilinear weights for nn.ConvTranspose

    weight_init_bilinear_(m_ConvTranspose) --> None

    Args:

        m_ConvTranspose: module with type of nn.ConvTranspose[1d/2d/3d]

    Examples:
    
    >>> m = nn.ConvTranspose2d(1, 1, 3)
    >>> weight_init_bilinear_(m)
    >>> m.weight.data
    tensor([[[[0.2500, 0.5000, 0.2500],
              [0.5000, 1.0000, 0.5000],
              [0.2500, 0.5000, 0.2500]]]])
    '''

    in_channels = m_ConvTranspose.in_channels
    out_channels = m_ConvTranspose.out_channels
    kernel_size = m_ConvTranspose.kernel_size

    # creat filt of kernel_size
    dims = len(kernel_size)
    filters = []
    for i in range(dims):
        kz = kernel_size[i]
        factor = (kz+1)//2
        center = factor-1.0 if(1==kz%2) else factor-0.5
        tfilter = torch.arange(kz).float()
        tfilter = 1 - (tfilter - center).abs()/factor
        filters.append(tfilter)
    
    # cross multiply filters
    filter = filters[0]
    for i in range(1, dims):
        filter = filter[:, None] * filters[i]
    filter = filter.type_as(m_ConvTranspose.weight.data)

    # fill filt for diagonal line
    channels = min(in_channels, out_channels)
    for i, j in zip(range(channels), range(channels)):
        m_ConvTranspose.weight.data[i, j][:] = filter


#----------- upsample and imwrap ---------------------------------------------#
def upsample_as_bilinear(in_tensor, ref_tensor):

    rh, rw = ref_tensor.shape[-2:]
    ih, iw = in_tensor.shape[-2:]
    assert (rh >= ih) and (rw >= iw), str([ih, iw, rh, rw])
    out_tensor = F.interpolate(in_tensor, size=(rh, rw), 
                        mode='bilinear', align_corners = True)
    return out_tensor


def upsample_as_nearest(in_tensor, ref_tensor):

    rh, rw = ref_tensor.shape[-2:]
    ih, iw = in_tensor.shape[-2:]
    assert (rh >= ih) and (rw >= iw), str([ih, iw, rh, rw])
    out_tensor = F.interpolate(in_tensor, size=(rh, rw), mode='nearest')
    return out_tensor


def imwrap(imR, dispL_norm, rect={'xs': -1, 'xe':1, 'ys':-1, 'ye':1}):
    '''
    Wrap right image to left view according to normal left disparity 
    
    imwrap(imR, dispL_norm, rect={'xs': -1, 'xe':1, 'ys':-1, 'ye':1}) --> imL_wrap
    
    Args:

        imR: the right image, with shape of [bn, c , h0, w0]
        dispL_norm: normal left disparity, with shape of [bn, 1 , h, w]
        rect: the area of left image for the dispL_norm, 
              consist the keys of ['xs', 'xe', 'ys', 'ye'].
              'xs': start position of width direction,
              'xe': end position of width direction,
              'ys': start of height direction,
              'ye': end of height direction,
              such as rect={'xs': -1, 'xe':1, 'ys':-1, 'ye':1} for all area.
    
    Examples:

        >>> imR = torch.rand(1, 3, 32, 32)
        >>> dispL = torch.ones(1, 1, 16, 16)*0.1
        >>> rect = {'xs': -0.5, 'xe':0.5, 'ys':-0.5, 'ye':0.5}
        >>> w_imL = imwrap(imR, dispL, rect)
        >>> w_imL.shape[-2:] == dispL.shape[-2:]
        True
    '''

    # get shape of dispL_norm
    bn, c, h, w = dispL_norm.shape
    
    # create sample grid
    row = torch.linspace(rect['xs'], rect['xe'], w)
    col = torch.linspace(rect['ys'], rect['ye'], h)
    grid_x = row[:, None].expand(bn, h, w, 1)
    grid_y = col[:, None, None].expand(bn, h, w, 1)
    grid = torch.cat([grid_x, grid_y], dim=-1).type_as(dispL_norm)
    grid[..., 0] = (grid[..., 0] - dispL_norm.squeeze(1))
    
    # sample image by grid
    imL_wrap = F.grid_sample(imR, grid)
    
    # refill 0 for out-of-bound grid locations
    mask = (grid[..., 0]<-1)|(grid[..., 0]>1)
    imL_wrap[mask[:, None].expand_as(imL_wrap)] = 0
    
    # visualize weight
    if(visualize_wraped): # and not self.training):
    
        imgs = imR[0, :3].permute(1, 2, 0).squeeze(-1)
        plt.subplot(221); plt.imshow(imgs.data.cpu().numpy())
        imgs = imL_wrap[0, :3].permute(1, 2, 0).squeeze(-1)
        plt.subplot(223); plt.imshow(imgs.data.cpu().numpy())
        
        imgs = dispL_norm[0, 0]
        plt.subplot(222); plt.imshow(imgs.data.cpu().numpy())
        imgs = grid[0, :, :, 0]
        plt.subplot(224); plt.imshow(imgs.data.cpu().numpy())
        plt.show()

    return imL_wrap


def normalize(tensor):

    bn = tensor.size(0)
    tmin, _ = tensor.view(bn, -1).min(dim=1)
    tmax, _ = tensor.view(bn, -1).max(dim=1)

    shape = [bn] + [1]*(tensor.dim()-1)
    tmin, tmax = tmin.view(shape), tmax.view(shape)
    tensor = (tensor - tmin)/(tmax-tmin).clamp(1e-8)

    return tensor
    
    
#----------- disp_volume_gen and disp_regression -----------------------------#
def disp_volume_gen(fL, fR, shift, stride=1):
    """
    generate 5D volume by concatenating 4D left feature and shift 4D right feature

    disp_volume_gen(fL, fR, shift, stride=1) --> 5D disp_volume

    Args:

        fL: 4D left feature
        fR: 4D right feature
        shift: count of shift 4D right feature
        stride: stride of shift 4D right feature

    Examples:
    
        >>> fL = torch.rand(2, 16, 9, 9)
        >>> fR = torch.rand(2, 16, 9, 9)
        >>> y = disp_volume_gen(fL, fR, 4, 2)
        >>> list(y.shape)
        [2, 32, 4, 9, 9]
    """

    bn, c, h, w = fL.shape
    cost = torch.zeros(bn, c*2, shift,  h,  w).type_as(fL.data)
    for i in range(0, shift):
        idx = i*stride
        cost[:, :c, i, :, idx:] = fL[..., idx:]
        cost[:, c:, i, :, idx:] = fR[..., :w-idx]
    return cost.contiguous()


def disp_regression(similarity, step_disp):
    """
    Returns predicted disparity with argsofmax(disp_similarity).

    disp_regression(similarity, step_disp) --> tensor[disp]

    Predicted disparity is computed as: d_predicted = sum_d( d * P_predicted(d))
    
    Args:

        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        step_disp: disparity difference between near-by
                   disparity indices in "similarities" tensor.

    Examples:
    
        >>> x = torch.rand(2, 20, 2, 2)
        >>> y = disp_regression(x, 1)
        >>> 0 < y.max().item() < 20
        True
    """    

    assert 4 == similarity.dim(), \
    'Similarity should 4D Tensor,but get {}D Tensor'.format(similarity.dim())
    
    P = F.softmax(similarity, dim=1)
    disps = torch.arange(0, P.size(-3)).type_as(P.data)*step_disp
    return torch.sum(P*disps[None, :, None, None], 1, keepdim=True)
    

def disp_regression_dw(similarity, dw_vol, step_disp):
    """
    Returns predicted disparity with argsofmax(disp_similarity).

    disp_regression(similarity, step_disp) --> tensor[disp]

    Predicted disparity is computed as: d_predicted = sum_d( d * P_predicted(d))
    
    Args:

        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        dw_vol: Tensor with delt width with indices
                     [example_index, disparity_index, y, x].
        step_disp: disparity difference between near-by
                   disparity indices in "similarities" tensor.

    Examples:
    
        >>> x = torch.rand(2, 20, 2, 2)
        >>> y = disp_regression(x, 1)
        >>> 0 < y.max().item() < 20
        True
    """    

    assert 4 == similarity.dim(), \
    'Similarity should 4D Tensor,but get {}D Tensor'.format(similarity.dim())
    
    P = F.softmax(similarity, dim=1)
    disps = torch.arange(0, P.size(-3)).type_as(P.data)*step_disp
    disps = disps[None, :, None, None] + dw_vol

    return torch.sum(P*disps, 1, keepdim=True)
    

def disp_regression_nearby(similarity, step_disp, half_support_window=2):
    """
    Returns predicted disparity with subpixel_map(disp_similarity).

    disp_regression_nearby(similarity, step_disp, half_support_window=2) --> tensor[disp]

    Predicted disparity is computed as: 
    
    d_predicted = sum_d( d * P_predicted(d)), 
    where | d - d_similarity_maximum | < half_size

    Args:

        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        step_disp: disparity difference between near-by
                   disparity indices in "similarities" tensor.
        half_support_window: defines size of disparity window in pixels
                             around disparity with maximum similarity,
                             which is used to convert similarities
                             to probabilities and compute mean.

    Examples:
    
        >>> x = torch.rand(2, 20, 2, 2)
        >>> y = disp_regression_nearby(x, 1).view(-1)
        >>> 0 < y.max().item() < 20
        True
    """    

    assert 4 == similarity.dim(), \
    'Similarity should 4D Tensor,but get {}D Tensor'.format(similarity.dim())

    # In every location (x, y) find disparity with maximum similarity score.
    similar_maximum, idx_maximum = torch.max(similarity, dim=1, keepdim=True)
    idx_limit = similarity.size(1)-1
    
    # Collect similarity scores for the disparities around the disparity
    # with the maximum similarity score.
    support_idx_disp = []
    for idx_shift in range(-half_support_window, half_support_window+1):

        idx_disp = idx_maximum + idx_shift
        idx_disp[idx_disp < 0] = 0
        idx_disp[idx_disp >= idx_limit] = idx_limit
        support_idx_disp.append(idx_disp)

    support_idx_disp = torch.cat(support_idx_disp, dim=1) 
    support_similar = torch.gather(similarity, 1, support_idx_disp.long())
    support_disp = support_idx_disp.float()*step_disp

    # Convert collected similarity scores to the disparity distribution
    # using softmax and compute disparity as a mean of this distribution.
    prob = F.softmax(support_similar, dim=1)
    disp = torch.sum(prob * support_disp.float(), dim=1, keepdim=True)

    return disp
    

#----------- doctest --------------------------------------------------------#
if __name__ == '__main__':

    import doctest
    doctest.testmod()



