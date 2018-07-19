# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import config as cfg
import numpy as np
from skimage.measure import block_reduce as pool_m
from numpy.matlib import repmat

def __make_matrix__(T, H, W):
    n = T*H*W
    M = np.zeros((n,n,27)).astype('float32')
    for i in range(n):
        t = int(i/H/W)
        h = int(i/W) - t*H
        w = i - t*H*W - h*W
        k = 0
        for tt in range(t-1,t+2):
            for hh in range(h-1,h+2):
                for ww in range(w-1,w+2):
                    if hh>-1 and hh<H and ww>-1 and ww<W and tt>-1 and tt<T:
                        M[i,tt*H*W+hh*W+ww,k] = 1
                    k += 1
    return M

def spacetime_nonlocal(
        model, blob_in, dim_in, dim_out, batch_size, 
        pool_stride, height, width,
        prefix, dim_inner,
        is_test, max_pool_stride=2):
    rel = model.ConvNd(
        blob_in, prefix + '_rel1',
        dim_in,
        dim_inner,
        [1, 5, 5],
        strides=[1, 1, 1],
        pads=[0, 2, 2] * 2,
        weight_init=('GaussianFill', {'std': cfg.NONLOCAL.CONV_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=cfg.NONLOCAL.NO_BIAS)

    rel = model.ConvNd(
        rel, prefix + '_rel2',
        dim_inner,
        27,
        [3, 1, 1],
        strides=[1, 1, 1],
        pads=[1, 0, 0] * 2,
        weight_init=('GaussianFill', {'std': cfg.NONLOCAL.CONV_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=cfg.NONLOCAL.NO_BIAS)
    rel = model.SpatialBN(
                rel, prefix + "_bn", dim_in,
                epsilon=cfg.NONLOCAL.BN_EPSILON, momentum=cfg.NONLOCAL.BN_MOMENTUM,
                is_test=is_test
            )
    model.param_init_net.ConstantFill(
                [prefix + "_bn_s"], prefix + "_bn_s", value=cfg.NONLOCAL.BN_INIT_GAMMA) 

    rel, rel_3d1 = model.net.Reshape(
        rel, [rel + '_re1' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else rel, rel + '_shape3d1'],
        shape=(batch_size, 27, -1)) # batch 27 THW
    rel = model.Transpose(rel, rel + '_trans', axes=(0, 2, 1))# batch THW 27
    rel, rel_3d2 = model.net.Reshape(
        rel, [rel + '_re2' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else rel, rel + '_shape3d2'],
        shape=(batch_size, -1, 27, 1)) ##> batch THW 27 1
    rel = model.Softmax(rel, rel + '_softmax', engine='CUDNN', axis=2) 

    _sparse_ = __make_matrix__(batch_size, pool_stride, height, width) ##> batch THW THW 27
    sparser = model.net.GivenTensorFill([], prefix + "sparser", shape=_sparse_.shape, values=_sparse_)
    sparse = model.net.BatchMatMul([sparser, rel], prefix + '_sparse') ## batch THW THW 1
    sparse, sparse_3d = model.net.Reshape(
        sparse, [sparse + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else sparse, sparse + '_shape'],
        shape=(batch_size, pool_stride*height*width, pool_stride*height*width)) ##>> batch, THW, THW

    g, g_shape_5d = model.net.Reshape(
        g, [g + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else g, g + '_shape5d'],
        shape=(batch_size, dim_in, -1)) ##>> batch dim_in THW

    t = model.net.BatchMatMul([g, sparse], prefix + '_y', trans_b=1) ## batch dim_in THW
    blob_out, t_shape = model.net.Reshape(
        t, [t + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else t, t+'_shape3d'],
        shape=(batch_size, dim_in, pool_stride, height, width))
    return blob_out

def add_nonlocal(model, blob_in, dim_in, dim_out, batch_size, 
                 pool_stride, height, width, 
                 prefix, dim_inner):
    is_test = model.split in ['test', 'val']
    blob_out = spacetime_nonlocal(
        model, blob_in, dim_in, dim_out, batch_size, 
        pool_stride, height, width, 
        prefix, dim_inner, is_test)
    blob_out = model.net.Sum([blob_in, blob_out], prefix + "_sum")
    return blob_out


# this is to reduce memory usage if the feature maps are big
# divide the feature maps into groups in the temporal dimension,
# and perform Non-local operations inside each group
def add_nonlocal_group(
        model, blob_in, dim_in, dim_out, batch_size, pool_stride, height, width,
        group_size, prefix, dim_inner):
    is_test = model.split in ['test', 'val']

    group_num = int(pool_stride / group_size)
    assert(pool_stride % group_size == 0)

    if group_num > 1:
        blob_in = model.Transpose(blob_in, blob_in + '_trans', axes=(0, 2, 1, 3, 4))
        blob_in, blob_in_5d = model.Reshape(
            blob_in, [blob_in + '_re'
            if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else blob_in,
            blob_in + '_shape5d'],
            shape=(batch_size * group_num, group_size, dim_in, height, width))
        blob_in = model.Transpose(blob_in, blob_in + '_trans', axes=(0, 2, 1, 3, 4))

    blob_out = spacetime_nonlocal(
        model, blob_in, dim_in, dim_out, batch_size * group_num,
        group_size, height, width,
        prefix, dim_inner, is_test)
    blob_out = model.net.Sum([blob_in, blob_out], prefix + "_sum")

    if group_num > 1:
        blob_out = model.Transpose(blob_out, blob_out + '_trans', axes=(0, 2, 1, 3, 4))
        blob_out, blob_out_shape = model.Reshape(
            [blob_out, blob_in_5d],
            [blob_out + '_re'
            if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else blob_out,
            blob_out + '_shape5d'])
        blob_out = model.Transpose(blob_out, blob_out + '_trans', axes=(0, 2, 1, 3, 4))

    return blob_out
