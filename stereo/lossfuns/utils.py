#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


visualize_wraped = False # True # 

to_numpy = lambda tensor: tensor.data.cpu().numpy()

diff1_dw = lambda tensor: F.pad(tensor[..., :, 1:]-tensor[..., :, :-1], [0, 1, 0, 0])
diff1_dh = lambda tensor: F.pad(tensor[..., 1:, :]-tensor[..., :-1, :], [0, 0, 0, 1])
diff1_dc1 = lambda tensor: F.pad(tensor[..., 1:, 1:]-tensor[..., :-1, :-1], [0, 1, 0, 1])
diff1_dc2 = lambda tensor: F.pad(tensor[..., 1:, :-1]-tensor[..., :-1, 1:], [1, 0, 0, 1])

diff2_dw = lambda tensor: F.pad(tensor[..., :, :-2] + tensor[..., :, 2:] - 2.0*tensor[..., :, 1:-1], [1, 1, 0, 0])
diff2_dh = lambda tensor: F.pad(tensor[..., :-2, :] + tensor[..., 2:, :] - 2.0*tensor[..., 1:-1, :], [0, 0, 1, 1])
diff2_dc1 = lambda tensor: F.pad(tensor[..., :-2, :-2] + tensor[..., 2:, 2:] - 2.0*tensor[..., 1:-1, 1:-1], [1, 1, 1, 1])
diff2_dc2 = lambda tensor: F.pad(tensor[..., :-2, 2:] + tensor[..., 2:, :-2] - 2.0*tensor[..., 1:-1, 1:-1], [1, 1, 1, 1])

def diff2_norm(disp, direction=0):
    k = 1.0/disp.clamp(0.1)
    if(0==direction): 
        return F.pad(k[...,:,1:-1]*(disp[..., :, :-2] + disp[..., :, 2:]) - 2, [1, 1, 0, 0])
    elif(1==direction): 
        return F.pad(k[...,1:-1,:]*(disp[..., :-2, :] + disp[..., 2:, :]) - 2, [0, 0, 1, 1])
    elif(2==direction): 
        return F.pad(k[...,1:-1,1:-1]*(disp[..., :-2, :-2] + disp[..., 2:, 2:]) - 2, [1, 1, 1, 1])
    elif(3==direction): 
        return F.pad(k[...,1:-1,1:-1]*(disp[..., :-2, 2:] + disp[..., 2:, :-2]) - 2, [1, 1, 1, 1])
    else: raise Exception('no supported direction for difference[diff2_norm]')
    
    
def diff_z_norm(disp, direction=0):
    k = 1.0/disp.clamp(0.1)
    if(0==direction): 
        return F.pad(disp[...,:,1:-1]*(k[..., :, :-2] + k[..., :, 2:]) - 2, [1, 1, 0, 0])
    elif(1==direction): 
        return F.pad(disp[...,1:-1,:]*(k[..., :-2, :] + k[..., 2:, :]) - 2, [0, 0, 1, 1])
    elif(2==direction): 
        return F.pad(disp[...,1:-1,1:-1]*(k[..., :-2, :-2] + k[..., 2:, 2:]) - 2, [1, 1, 1, 1])
    elif(3==direction): 
        return F.pad(disp[...,1:-1,1:-1]*(k[..., :-2, 2:] + k[..., 2:, :-2]) - 2, [1, 1, 1, 1])
    else: raise Exception('no supported direction for difference[diff_z_norm]')
    
    
diff3_dw = lambda disp: diff2_norm(disp, 0)
diff3_dh = lambda disp: diff2_norm(disp, 1)
diff3_dc1 = lambda disp: diff2_norm(disp, 2)
diff3_dc2 = lambda disp: diff2_norm(disp, 3)

diff4_dw = lambda disp: diff_z_norm(disp, 0)
diff4_dh = lambda disp: diff_z_norm(disp, 1)
diff4_dc1 = lambda disp: diff_z_norm(disp, 2)
diff4_dc2 = lambda disp: diff_z_norm(disp, 3)


def imshow_tensor(img):
    '''
    使用matplotlib.pyplot绘制单个tensor类型图片
    图片尺寸应为(1, c, h, w)
    '''
    if img is None:
        plt.cla()
    else:
        _, c, h, w = img.shape
        if 3 <= c:
            plt.imshow(to_numpy(img[0]).transpose(1, 2, 0))
        else: 
            plt.imshow(to_numpy(img[0, 0]))


def plot_tensors(*tensors):
    """
    使用matplotlib.pyplot绘制多个tensor类型图片
    图片尺寸应为(bn, c, h, w)
    或是单个图片尺寸为(1, c, h, w)的序列
    """
    count = min(8, len(tensors))
    if(count==0): return
    col = min(2, count)
    row = count//col
    if(count%col > 0):
        row = row + 1
    for i in range(count):
        plt.subplot(row, col, i+1); imshow_tensor(tensors[i])
    
    return plt


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


class AvgPool2d_mask(nn.AvgPool2d):
    """
    Average filter over an input signal 

    AvgBlur2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:
            `kernel_size`
        padding: implicit zero padding to be added on both sides
        ceil_mode: when True, will use `ceil` instead of `floor` 
            to compute the output shape. Default False.
        count_include_pad: when True, will include the zero-padding 
            in the averaging calculation. Default True.
   
    Examples:
    
        >>> # With square kernels and equal stride
        >>> filter = AvgPool2d_mask(2, ceil_mode=True)
        >>> input = torch.randn(1, 4, 5, 5)
        >>> mask = (input > 0.5)
        >>> output = filter(input, mask)
        >>> list(output.shape)
        [1, 4, 3, 3]
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(AvgPool2d_mask, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)

        self.avgpool = super(AvgPool2d_mask, self).forward


    def forward(self, input, mask=None):
        
        if(mask is None):
            return self.avgpool(input)
        else:
            mask = mask.float()
            output = self.avgpool(input*mask)
            avg_mask = self.avgpool(mask)
            return output/avg_mask.clamp(1e-8)


def filter2d(input, weight, stride=1, padding=0, dilation=1):
    """
    Applies a 2D filter over an input signal 

    filter2d(input, kernel, stride=1, padding=0, dilation=1) --> Tensor

    Args:

        input: input tensor of shape `(minibatch, in_channels, iH , iW)` 
        weight: filters of shape `(1 ,1 , kH , kW)`
        stride: the stride of the filter kernel. Can be a single number or a 
            tuple `(sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a single 
            number or a tuple `(padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number 
            or a tuple `(dH, dW)`. Default: 1
   
    Examples:
    
        >>> # With square kernels and equal stride
        >>> input = torch.randn(1, 4, 5, 5)
        >>> kernel = torch.randn(1,1,3,3)
        >>> output = filter2d(input, kernel, padding=1)
        >>> list(output.shape)
        [1, 4, 5, 5]
    """
    
    channel = input.shape[1]
    weight = weight.expand([channel] + list(weight.shape[-3:]))
    return F.conv2d(input, weight, stride=stride, padding=padding, 
                    groups=channel, dilation=dilation)


class GaussianBlur2d_linear(nn.Module):
    """
    Gaussian filter over an input signal 

    GaussianBlur2d(kernel_size=7, sigma=1.0, stride=1, padding=0, dilation=1)

    Args:

        kernel_size: the size of the filter kernel. Can be a single number 
            or a tuple `(kH, kW)`.
        sigma: the sigma of the filter kernel. Can be a float number or a
            tuple `(sigmaH, sigmak)`.
        stride: the stride of the filter kernel. Can be a single number or 
            a tuple `(sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a single 
            number or a tuple `(padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number 
            or a tuple `(dH, dW)`. Default: 1
   
    Examples:
    
        >>> # With square kernels and equal stride
        >>> filter = GaussianBlur2d_linear(3, padding=1)
        >>> input = torch.randn(1, 4, 5, 5)
        >>> output = filter(input)
        >>> list(output.shape)
        [1, 4, 5, 5]
    """

    def __init__(self, kernel_size=7, sigma=1.0, stride=1, padding=0, dilation=1):
        super(GaussianBlur2d_linear, self).__init__()

        fun_is_number = lambda x: isinstance(x, (int, float))
        fun_two_number = lambda x: (x, x) if fun_is_number(x) else x[:2]

        # key parameters
        self.kernel_size = fun_two_number(kernel_size)
        self.sigma = fun_two_number(sigma)
        self.stride = fun_two_number(stride)
        self.padding = fun_two_number(padding)
        self.dilation = fun_two_number(dilation)
        
        padh, padw = self.padding
        self.paddings = [padw, padw, padh, padh]
        
        # create kernel
        kH, kW = self.kernel_size
        sigmaH, sigmaW = self.sigma
        kernel_h = self.gaussion_kernel(kH, sigmaH)[None, None, :, None]
        kernel_w = self.gaussion_kernel(kW, sigmaW)[None, None, None, :]
        self.kernel_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.kernel_w = nn.Parameter(data=kernel_w, requires_grad=False)

        # kargs_filter
        self.kargs_filter_h = {'stride': (self.stride[0], 1), 'dilation': (self.dilation[0], 1), }
        self.kargs_filter_w = {'stride': (1, self.stride[1]), 'dilation': (1, self.dilation[1]), }


    def gaussion_kernel(self, kernel_size, sigma):

        x = (torch.arange(0, kernel_size) - kernel_size//2).float()
        gauss = torch.exp(-x*x/(2.0*sigma*sigma))
        
        return gauss/gauss.sum()


    def forward(self, input):
        
        input = F.pad(input, self.paddings, mode='replicate')
        output = filter2d(input, self.kernel_h, **self.kargs_filter_h)
        output = filter2d(output, self.kernel_w, **self.kargs_filter_w)
        
        return output


GaussianBlur2d = GaussianBlur2d_linear
class GuideFilter2d(nn.Module):
    """
    Guide filter over an input signal 

    GuideFilter2d(kernel_size=3, eps=0.01)

    Args:

        kernel_size: the size of the filter kernel. Can be a single number 
            or a tuple `(kH, kW)`.
        eps: adjust the corr factor of guide filter

    Examples:
    
        >>> # With square kernels and equal stride
        >>> filter = GuideFilter2d(3, 0.01)
        >>> input = torch.randn(1, 4, 5, 5)
        >>> output = filter(input)
        >>> list(output.shape)
        [1, 4, 5, 5]
    """

    def __init__(self, kernel_size=3, eps=0.01):
        super(GuideFilter2d, self).__init__()

        self.kernel_size = kernel_size
        self.eps = eps
        self.blur = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)


    def forward(self, input, guide=None):
        
        I, p = input, guide

        if(guide is not None):
            
            mean_I = self.blur(I)
            mean_p = self.blur(p)

            mean_II = self.blur(I * I)
            mean_Ip = self.blur(I * p)

            var_I = mean_II - mean_I * mean_I  # 方差
            cov_Ip = mean_Ip - mean_I * mean_p # 协方差

            a = cov_Ip / (var_I + self.eps)    # 相关因子a
            b = mean_p - a * mean_I            # 相关因子b
        
        else:

            mean_I = self.blur(I)
            mean_p = mean_I

            mean_II = self.blur(I * I)
            mean_Ip = mean_II

            var_I = mean_II - mean_I * mean_I  # 方差
            cov_Ip = var_I

            a = cov_Ip / (var_I + self.eps)    # 相关因子a
            b = mean_p - a * mean_I            # 相关因子b

        mean_a = self.blur(a)      # 对a进行均值平滑
        mean_b = self.blur(b)      # 对b进行均值平滑

        return mean_a * I + mean_b


class GaussionPyramid(nn.Module):
    """
    Gaussian pyramid

    GaussionPyramid(self, octaves=1, noctave=1, sigma=1.0)

    Args:

        octaves: the count of the octave group. Should be a single number. 
            Default: 1
        noctave: the image count in a octave group. Should be a single 
            number. Default: 1
        sigma: the sigma of the first filter. Should be a float number. 
            Default: 1.0
        noctave_ex: the image count with (simga>2*simga0) in a octave 
            group. Should be a single number. Default: 1
   
    Examples:
    
        >>> # With square kernels and equal stride
        >>> filter = GaussionPyramid(octaves=3, noctave=2, sigma=1.0, noctave_ex=1)
        >>> input = torch.randn(2, 4, 5, 5)
        >>> output = filter(input, normalize=True)
        >>> list([list(t.shape) for t in output])
        [[2, 16, 5, 5], [2, 16, 3, 3], [2, 16, 2, 2]]
    """

    def __init__(self, octaves=1, noctave=1, sigma=1.0, noctave_ex=0):
        super(GaussionPyramid, self).__init__()

        # key parameters
        self.octaves = max(1, octaves)
        self.noctave = max(1, noctave)
        self.sigma = sigma
        self.noctave_ex = max(0, noctave_ex)
        self._noctave = self.noctave + self.noctave_ex
        
        # sigmas of gaussion filter
        k0 = 2.0**(1.0/noctave)
        k1 = (k0*k0 - 1)**0.5
        sigmas = [k1*sigma]
        for i in range(self._noctave):
            sigmas.append(sigmas[-1]*k0)

        # groups of gaussion filter
        self.filters, self.paddings = [], []
        for i in range(self._noctave):
            padding = round(3*sigmas[i]+0.5) # ; print(sigmas[i], padding)
            kernel_size = padding*2 + 1
            filter = GaussianBlur2d(kernel_size, sigmas[i])
            self.paddings.append([padding, padding, padding, padding])
            self.filters.append(filter)
        self.filters = nn.ModuleList(self.filters)


    def forward(self, input, octaves=1, normalize=False):
        
        # gaussion pyramid
        bn = input.size(0)
        timg = input
        output = []
        for i in range(octaves):
            if(i > 0):
                timg = timg[..., ::2, ::2]
            timg1 = timg # current scale image
            imgs_octave = [timg1]
            for j in range(self._noctave):
                timg1 = F.pad(timg1, self.paddings[j], mode='replicate')
                timg1 = self.filters[j](timg1) # next scale image
                imgs_octave.append(timg1)
                if(j == self.noctave-1):
                    timg = timg1.clone() # the base image of the next octave
            imgs_octave = torch.cat(imgs_octave, dim=1)
            if(normalize):
                tmin, _ = imgs_octave.view(bn, -1).min(dim=1)
                tmax, _ = imgs_octave.view(bn, -1).max(dim=1)
                tmin, tmax = tmin.view(bn, 1, 1, 1), tmax.view(bn, 1, 1, 1)
                imgs_octave = (imgs_octave - tmin)/(tmax-tmin).clamp(1e-8)
            output.append(imgs_octave)        
        
        return output


class DoGPyramid(nn.Module):
    """
    Difference pyramid of gaussian 

    GaussionPyramid(self, octaves=1, noctave=1, sigma=1.0)

    Args:

        octaves: the count of the octave group. Should be a single number. 
            Default: 1
        noctave: the image count in a octave group. Should be a single 
            number. Default: 1
        sigma: the sigma of the first filter. Should be a float number. 
            Default: 1.0
        noctave_ex: the image count with (simga>2*simga0) in a octave 
            group. Should be a single number. Default: 1
   
    Examples:
    
        >>> # With square kernels and equal stride
        >>> filter = DoGPyramid(octaves=3, noctave=2, sigma=1.0, noctave_ex=1)
        >>> input = torch.randn(2, 4, 5, 5)
        >>> output = filter(input, normalize=True)
        >>> list([list(t.shape) for t in output])
        [[2, 16, 5, 5], [2, 16, 3, 3], [2, 16, 2, 2]]
    """

    def __init__(self, octaves=1, noctave=1, sigma=1.0, noctave_ex=0):
        super(DoGPyramid, self).__init__()

        # key parameters
        self.octaves = max(1, octaves)
        self.noctave = max(1, noctave)
        self.sigma = sigma
        self.noctave_ex = max(0, noctave_ex)
        self._noctave = self.noctave + self.noctave_ex
        
        # sigmas of gaussion filter
        k0 = 2.0**(1.0/noctave)
        k1 = (k0*k0 - 1)**0.5
        sigmas = [k1*sigma]
        for i in range(self._noctave + 1):
            sigmas.append(sigmas[-1]*k0)

        # groups of gaussion filter
        self.filters, self.paddings = [], []
        for i in range(self._noctave + 1):
            padding = round(3*sigmas[i]+0.5) # ; print(sigmas[i], padding)
            kernel_size = padding*2 + 1
            filter = GaussianBlur2d(kernel_size, sigmas[i])
            self.paddings.append([padding, padding, padding, padding])
            self.filters.append(filter)
        self.filters = nn.ModuleList(self.filters)


    def forward(self, input, octaves=1, normalize=False):
        
        # Difference pyramid of gaussian 
        bn = input.size(0)
        timg = input
        output = []
        for i in range(octaves):
            if(i > 0):
                timg = timg[..., ::2, ::2]
            timg1 = timg # current scale image
            imgs_octave = []
            for j in range(self._noctave+1):
                timg2 = F.pad(timg1, self.paddings[j], mode='replicate')
                timg2 = self.filters[j](timg2) # next scale image
                imgs_octave.append(timg2 - timg1)
                timg1 = timg2
                if(j == self.noctave-1):
                    timg = timg1.clone() # the base image of the next octave
            imgs_octave = torch.cat(imgs_octave, dim=1)
            if(normalize):
                tmin, _ = imgs_octave.view(bn, -1).min(dim=1)
                tmax, _ = imgs_octave.view(bn, -1).max(dim=1)
                tmin, tmax = tmin.view(bn, 1, 1, 1), tmax.view(bn, 1, 1, 1)
                imgs_octave = (imgs_octave - tmin)/(tmax-tmin).clamp(1e-8)
            output.append(imgs_octave)        

        return output


class SSIM(nn.Module):
    """
    Structural similarity(SSIM) index measurement with gaussian filter

    SSIM(kernel_size=7, sigma=1, padding=0)

    Args:

        kernel_size: the size of the gaussian filter kernel. Can be a 
            single number or a tuple `(kH, kW)`.
        sigma: the sigma of the gaussion filter kernel. Can be a float 
            number or a tuple `(sigmaH, sigmak)`.
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
   
    Examples::
    
        >>> input1 = torch.rand(1, 3, 16, 16)
        >>> input2 = torch.rand(1, 3, 16, 16)
        >>> ssim = SSIM(kernel_size=7, sigma=1, padding=3)
        >>> output = ssim(input1, input2)
        >>> 0 < output.mean().item() < 1
        True
    """

    
    def __init__(self, kernel_size=7, sigma=1, padding=0):
        super(SSIM, self).__init__()
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = padding
        self.filter = GaussianBlur2d(kernel_size, sigma, padding=padding)


    def forward(self, img1, img2):
        
        mu1 = self.filter(img1)
        mu2 = self.filter(img2)

        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2

        sigma1_sq = self.filter(img1*img1) - mu1_sq
        sigma2_sq = self.filter(img2*img2) - mu2_sq
        sigma12 = self.filter(img1*img2) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_n = (2*mu1_mu2 + C1)*(2*sigma12 + C2)
        ssim_d = (mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)
        ssim_map = ssim_n/ssim_d
        
        return ssim_map


def loss_disp(pred, target, mask=None, k=1.0):

    if(1.0 == k):
        return F.smooth_l1_loss(pred[mask], target[mask], reduction='mean')
    else:
        k = min(10, max(0.1, k))
        tk = 1.0/k
        return tk*F.smooth_l1_loss(k*pred[mask], k*target[mask], reduction='mean')


def _laplace_probability(value, center, diversity=1.0):
    return torch.exp(-torch.abs(center - value) / diversity) # / (2 * diversity)


def loss_subpixel_cross_entropy(similarity, disp_true, disp_step, diversity=1.0, mask=None, weights=None):
    """Returns sub-pixel cross-entropy loss.

    Cross-entropy is computed as

    - sum_d [ log(P_predicted(d)) * P_target(d) ]
      -------------------------------------------------
                        sum_d P_target(d)

    We need to normalize the cross-entropy by sum_d P_target(d),
    since the target distribution is not normalized.

    Args:
        similarity: Tensor with similarities with indices
                     [example_index, disparity_index, y, x].
        disp_true: Tensor with ground truth disparities with
                    indices [example_index, y, x]. The
                    disparity values are floats. The locations with unknown
                    disparities are filled with 'inf's.
        disp_step: disparity difference between near-by
                   disparity indices in "similarities" tensor.
        diversity: diversity of the target Laplace distribution,
                   centered at the sub-pixel ground truth.
        weights  : Tensor with weights of individual locations.
    """

    log_P_predicted = F.log_softmax(similarity, dim=1)
    sum_P_target = torch.zeros_like(disp_true)
    sum_P_target_x_log_P_predicted = torch.zeros_like(disp_true)

    for idx_disp in range(similarity.size(-3)):
        disparity = idx_disp * disp_step
        P_target = _laplace_probability(disparity, disp_true, diversity)
        sum_P_target += P_target
        sum_P_target_x_log_P_predicted += (log_P_predicted[:, idx_disp] * P_target)
    
    entropy = -sum_P_target_x_log_P_predicted[mask] / sum_P_target[mask]
    
    if weights is not None:
        weights = weights[mask]
        return (weights * entropy).sum() / (weights.sum() + 1e-15)

    return entropy.mean()


def loss_img_diff1(img, img_wrap):

    L1_dw = torch.abs(diff1_dw(img) - diff1_dw(img_wrap))
    L1_dh = torch.abs(diff1_dh(img) - diff1_dh(img_wrap))
    return L1_dw + L1_dh


def loss_disp_smooth(img, disp, mode='s2', flag_ad=False):
    
    assert mode in ['s1', 's2', 's3', 's4']

    bn = disp.size(0)

    def loss_one_path(fun_diff_im, fun_diff_disp):
        diff_im = fun_diff_im(img).abs().max(dim=1, keepdim=True)[0]
        if(flag_ad):
            mdiff_im = diff_im.view(bn, -1).mean(dim=-1).view(bn, 1, 1, 1).clamp(1e-8)
            diff_im /= 2.0*mdiff_im
            weight = torch.exp(-diff_im) + 0.01
        else:
            weight = torch.exp(-diff_im)
        diff_disp = fun_diff_disp(disp).abs()

        #plot_tensors(img, disp, weight, diff_disp)
        #plt.show()
        return weight*diff_disp

    # funs_diff_im
    if('s1' == mode):
        funs_diff_im = [diff1_dw, diff1_dh] #, diff1_dc1, diff1_dc2]
    else:
        funs_diff_im = [diff2_dw, diff2_dh] #, diff2_dc1, diff2_dc2]

    # funs_diff_disp
    if('s1' == mode):
        funs_diff_disp = [diff1_dw, diff1_dh] #, diff1_dc1, diff1_dc2]
    elif('s2' == mode):
        funs_diff_disp = [diff2_dw, diff2_dh] #, diff2_dc1, diff2_dc2]
    elif('s3' == mode):
        funs_diff_disp = [diff3_dw, diff3_dh] #, diff3_dc1, diff3_dc2]
    elif('s4' == mode):
        funs_diff_disp = [diff4_dw, diff4_dh] #, diff4_dc1, diff4_dc2]
    else:
        raise Exception('Not supported mode of loss_disp_smooth: %s' % mode)

    return 0.5*sum(map(loss_one_path, funs_diff_im, funs_diff_disp))


def loss_disp_smooth_edge(disp, edge, mode='s2'):
    
    # funs_diff_disp
    if('s1' == mode):
        funs_diff_disp = [diff1_dw, diff1_dh] #, diff1_dc1, diff1_dc2]
    elif('s2' == mode):
        funs_diff_disp = [diff2_dw, diff2_dh] #, diff2_dc1, diff2_dc2]
    elif('s3' == mode):
        funs_diff_disp = [diff3_dw, diff3_dh] #, diff3_dc1, diff3_dc2]
    elif('s4' == mode):
        funs_diff_disp = [diff4_dw, diff4_dh] #, diff4_dc1, diff4_dc2]
    else:
        raise Exception('Not supported mode of loss_disp_smooth: %s' % mode)

    # loss
    loss_one_path = lambda fun_diff_disp: fun_diff_disp(disp).abs()
    loss_ds = torch.cat(list(map(loss_one_path, funs_diff_disp)), dim=1)
    
    if(4 > edge.dim()):
        return 0.5*loss_ds.mean()

    else:

        bn = edge.size(0)
        mean_edge = edge.view(bn, -1).mean(-1).view([bn, 1, 1, 1]) + 0.01
        edge_inv = (1-edge).expand_as(loss_ds)
        loss_ds_edge = (loss_ds*edge_inv).mean() + (loss_ds*mean_edge).mean()

        return 0.5*loss_ds_edge


if __name__ == '__main__':

    import doctest
    doctest.testmod()
    


