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
"""DenseNetY, implemented in Gluon."""
__all__ = ['DenseNetY',
           'densenet13_y',  'densenet21_y',  'densenet37_y',  'densenet69_y']

import os

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity
from mxnet import base


# Helpers
def _make_dense_block(bits, bits_a, num_layers, bn_size, growth_rate, dropout, stage_index):
    out = nn.HybridSequential(prefix='stage%d_'%stage_index)
    with out.name_scope():
        for _ in range(num_layers):
            out.add(_make_dense_layer(bits, bits_a, growth_rate, bn_size, dropout))
    return out


def _make_dense_layer(bits, bits_a, growth_rate, bn_size, dropout):
    new_features = nn.HybridSequential(prefix='')
    if bn_size == 0:
        # no bottleneck
        new_features.add(nn.QActivation(bits=bits_a))
        new_features.add(nn.QConv2D(growth_rate, bits=bits, kernel_size=3, padding=1))
        if dropout:
            new_features.add(nn.Dropout(dropout))
        new_features.add(nn.BatchNorm())
    else:
        # bottleneck design
        new_features.add(nn.BatchNorm())
        new_features.add(nn.QActivation(bits=bits_a))
        new_features.add(nn.QConv2D(bn_size * growth_rate, bits=bits, kernel_size=1))
        if dropout:
            new_features.add(nn.Dropout(dropout))
        new_features.add(nn.BatchNorm())
        new_features.add(nn.QActivation(bits=bits_a))
        new_features.add(nn.QConv2D(growth_rate, bits=bits, kernel_size=3, padding=1))
        if dropout:
            new_features.add(nn.Dropout(dropout))

    out = HybridConcurrent(axis=1, prefix='')
    out.add(Identity())
    out.add(new_features)

    return out


def _make_transition(bits, bits_a, num_output_features):
    out = nn.HybridSequential(prefix='')
    out.add(nn.QActivation(bits=bits_a))
    out.add(nn.QConv2D(num_output_features, bits=bits, kernel_size=1))
    out.add(nn.AvgPool2D(pool_size=2, strides=2))
    out.add(nn.BatchNorm())
    return out

# Net
class DenseNetY(HybridBlock):
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
    """
    def __init__(self, bits, bits_a, num_init_features, growth_rate, block_config, reduction, bn_size,
                 modifier=[], thumbnail=False, dropout=0, classes=1000, **kwargs):
        assert len(modifier) == 0

        super(DenseNetY, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(nn.Conv2D(num_init_features, kernel_size=3, strides=1, padding=1, in_channels=0,
                                            use_bias=False))
            else:
                self.features.add(nn.Conv2D(num_init_features, kernel_size=7,
                                            strides=2, padding=3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
                self.features.add(nn.BatchNorm())
            # Add dense blocks
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                self.features.add(_make_dense_block(bits, bits_a, num_layers, bn_size, growth_rate, dropout, i+1))
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    features_after_transition = num_features // reduction[i]
                    # make it to be multiples of 32
                    features_after_transition = int(round(features_after_transition / 32)) * 32
                    self.features.add(_make_transition(bits, bits_a, features_after_transition))
                    num_features = features_after_transition
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.AvgPool2D(pool_size=4 if thumbnail else 7))
            self.features.add(nn.Flatten())

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Specification
# init_features, growth_rate, bn_size, reduction, block_config
densenet_y_spec = {
    13: (64, 32, 0, 1, [1, 1, 1, 1]),
    21: (64, 32, 0, 1, [2, 2, 2, 2]),
    37: (64, 32, 0, 1, [4, 4, 4, 4]),
    69: (64, 32, 0, 1, [8, 8, 8, 8]),
}


# Constructor
def get_densenet_y(num_layers, pretrained=False, ctx=cpu(), bits=1, bits_a=1,
                 opt_init_features=None, opt_growth_rate=None, opt_reduction=None,
                 root=os.path.join(base.data_dir(), 'models'), **kwargs):
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet_y. Options are 121, 161, 169, 201.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    init_features, growth_rate, bn_size, reduction, block_config = densenet_y_spec[num_layers]
    num_transition_blocks = len(block_config) - 1
    if opt_init_features is not None:
        init_features = opt_init_features
    if opt_growth_rate is not None:
        growth_rate = opt_growth_rate
    if opt_reduction is not None:
        split = [float(x) for x in opt_reduction.split(",")]
        if len(split) == 1:
            split *= num_transition_blocks
        reduction = split
        assert len(reduction) == num_transition_blocks, "need one or three values for --reduction"
    else:
        reduction = [reduction] * num_transition_blocks
    net = DenseNetY(bits, bits_a, init_features, growth_rate, block_config, reduction, bn_size, **kwargs)
    if pretrained:
        raise ValueError("No pretrained model exists, yet.")
        # from ..model_store import get_model_file
        # net.load_parameters(get_model_file('densenet_y%d'%(num_layers), root=root), ctx=ctx)
    return net

def densenet13_y(**kwargs):
    r"""Densenet-BC 13-layer model inspired by
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_densenet_y(13, **kwargs)

def densenet21_y(**kwargs):
    r"""Densenet-BC 21-layer model inspired by
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_densenet_y(21, **kwargs)

def densenet37_y(**kwargs):
    r"""Densenet-BC 37-layer model inspired by
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_densenet_y(37, **kwargs)

def densenet69_y(**kwargs):
    r"""Densenet-BC 69-layer model inspired by
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_densenet_y(69, **kwargs)
