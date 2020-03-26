# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""BaseNetDense, implemented in Gluon."""
import os

from mxnet import cpu, base

__all__ = ['BaseNetDense', 'BaseNetDenseParameters', 'get_basenet_constructor', 'DOWNSAMPLE_STRUCT']

import logging

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity

from .model_parameters import ModelParameters
from .common_layers import ChannelShuffle, add_initial_layers


DOWNSAMPLE_STRUCT = "bn,max_pool,relu,fp_conv"


# Net
class BaseNetDense(HybridBlock):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_init_features : int
        Number of filters to learn in the first convolution layer.
    growth_rate : int
        Number of filters to add each layer (`k` in the paper).
    block_config : list of int
        List of integers for numbers of layers in each pooling block.
    bn_size : int, default 4
        Multiplicative factor for number of bottle neck layers.
        (i.e. bn_size * k features in the bottleneck layer)
    dropout : float, default 0
        Rate of dropout after each dense layer.
    classes : int, default 1000
        Number of classification classes.
    initial_layers : bool, default imagenet
        Configure the initial layers.
    """

    def __init__(self, num_init_features, growth_rate, block_config, reduction, bn_size, downsample,
                 initial_layers="imagenet", dropout=0, classes=1000, dilated=False, **kwargs):
        super(BaseNetDense, self).__init__(**kwargs)
        self.num_blocks = len(block_config)
        self.dilation = (1, 1, 2, 4) if dilated else (1, 1, 1, 1)
        self.downsample_struct = downsample
        self.bn_size = bn_size
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.reduction_rates = reduction

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            add_initial_layers(initial_layers, self.features, num_init_features)
            # Add dense blocks
            self.num_features = num_init_features
            for i, repeat_num in enumerate(block_config):
                self._make_repeated_base_blocks(repeat_num, i)
                if i != len(block_config) - 1:
                    self._make_transition(i)
            self.finalize = nn.HybridSequential(prefix='')
            self.finalize.add(nn.BatchNorm())
            self.finalize.add(nn.Activation('relu'))
            if dilated:
                self.finalize.add(nn.AvgPool2D(pool_size=28))
            else:
                self.finalize.add(nn.AvgPool2D(pool_size=4 if initial_layers == "thumbnail" else 7))
            self.finalize.add(nn.Flatten())

            self.output = nn.Dense(classes)

    def get_layer(self, num):
        name = "layer{}".format(num)
        if not hasattr(self, name):
            setattr(self, name, nn.HybridSequential(prefix=''))
        return getattr(self, name)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        for i in range(self.num_blocks):
            x = self.get_layer(i)(x)
        x = self.finalize(x)
        x = self.output(x)
        return x

    def _add_base_block_structure(self, dilation):
        raise NotImplementedError()

    def _make_repeated_base_blocks(self, repeat_num, stage_index):
        dilation = self.dilation[stage_index]
        self.current_stage = nn.HybridSequential(prefix='stage{}_'.format(stage_index + 1))
        with self.current_stage.name_scope():
            for _ in range(repeat_num):
                self._add_base_block_structure(dilation)
        self.get_layer(stage_index).add(self.current_stage)

    def _add_dense_block(self, dilation):
        new_features = nn.HybridSequential(prefix='')

        def _add_conv_block(layer):
            new_features.add(nn.BatchNorm())
            new_features.add(layer)
            if self.dropout:
                new_features.add(nn.Dropout(self.dropout))

        if self.bn_size == 0:
            # no bottleneck
            _add_conv_block(nn.activated_conv(self.growth_rate, kernel_size=3, padding=dilation, dilation=dilation))
        else:
            # bottleneck design
            _add_conv_block(nn.activated_conv(self.bn_size * self.growth_rate, kernel_size=1))
            _add_conv_block(nn.activated_conv(self.growth_rate, kernel_size=3, padding=1))

        self.num_features += self.growth_rate

        dense_block = HybridConcurrent(axis=1, prefix='')
        dense_block.add(Identity())
        dense_block.add(new_features)
        self.current_stage.add(dense_block)

    def _make_transition(self, transition_num):
        dilation = self.dilation[transition_num + 1]
        num_out_features = self.num_features // self.reduction_rates[transition_num]
        num_out_features = int(round(num_out_features / 32)) * 32
        logging.info("Features in transition {}: {} -> {}".format(
            transition_num + 1, self.num_features, num_out_features
        ))
        self.num_features = num_out_features

        transition = nn.HybridSequential(prefix='')
        with transition.name_scope():
            for layer in self.downsample_struct.split(","):
                if layer == "bn":
                    transition.add(nn.BatchNorm())
                elif layer == "relu":
                    transition.add(nn.Activation("relu"))
                elif layer == "q_conv":
                    transition.add(nn.activated_conv(self.num_features, kernel_size=1))
                elif "fp_conv" in layer:
                    groups = 1
                    if ":" in layer:
                        groups = int(layer.split(":")[1])
                    transition.add(nn.Conv2D(self.num_features, kernel_size=1, groups=groups, use_bias=False))
                elif layer == "pool" and dilation == 1:
                    transition.add(nn.AvgPool2D(pool_size=2, strides=2))
                elif layer == "max_pool" and dilation == 1:
                    transition.add(nn.MaxPool2D(pool_size=2, strides=2))
                elif "cs" in layer:
                    groups = 16
                    if ":" in layer:
                        groups = int(layer.split(":")[1])
                    transition.add(ChannelShuffle(groups=groups))

        self.get_layer(transition_num + 1).add(transition)


class BaseNetDenseParameters(ModelParameters):
    def _is_it_this_model(self, model):
        raise NotImplementedError()

    def _map_opt_to_kwargs(self, opt, kwargs):
        kwargs['overwrite_reduction'] = opt.reduction
        kwargs['overwrite_growth_rate'] = opt.growth_rate
        kwargs['overwrite_init_features'] = opt.init_features
        kwargs['overwrite_downsample'] = opt.downsample_structure
        kwargs['dilated'] = opt.dilated
        if "flex" in opt.model:
            kwargs['flex_block_config'] = [int(x) for x in opt.block_config.split(",")]

    def _add_arguments(self, parser):
        parser.add_argument('--reduction', type=str, default=None,
                            help='divide channels by this number in transition blocks (3 values, e.g. "2,2.5,3")')
        parser.add_argument('--growth-rate', type=int, default=None,
                            help='add this many features each block')
        parser.add_argument('--init-features', type=int, default=None,
                            help='start with this many filters in the first layer')
        parser.add_argument('--downsample-structure', type=str, default=None,
                            help='layers in downsampling branch (available: bn,relu,conv,fp_conv,pool,max_pool)')
        parser.add_argument('--block-config', type=str, default=None,
                            help='how many blocks to use in a flex model')
        parser.add_argument('--dilated', action="store_true",
                            help='whether to use dilation, e.g. for segmentation')


def get_basenet_constructor(spec, net_constructor, default_bn_size=0, default_init_features=64, default_growth_rate=64):
    def constructor(num_layers, pretrained=False, ctx=cpu(), root=os.path.join(base.data_dir(), 'models'),
                    overwrite_init_features=None, overwrite_growth_rate=None, overwrite_downsample=None,
                    overwrite_reduction=None, flex_block_config=None, **kwargs):
        init_features = default_init_features if overwrite_init_features is None else overwrite_init_features
        growth_rate = default_growth_rate if overwrite_growth_rate is None else overwrite_growth_rate

        block_config, reduction_factor, downsample = spec[num_layers]
        reduction = [1 / x for x in reduction_factor]

        if num_layers is None:
            block_config = flex_block_config

        if overwrite_downsample is not None:
            downsample = overwrite_downsample

        num_transition_blocks = len(block_config) - 1
        if overwrite_reduction is not None:
            reduction = [float(x) for x in overwrite_reduction.split(",")]

        assert len(reduction) == num_transition_blocks, "need three values for --reduction"
        net = net_constructor(init_features, growth_rate, block_config, reduction, default_bn_size, downsample, **kwargs)
        if pretrained:
            raise ValueError("No pretrained model with automatic downloading exists, yet.")
            # from ..model_store import get_model_file
            # net.load_parameters(get_model_file('densenet%d'%(num_layers), root=root), ctx=ctx)
        return net
    return constructor
