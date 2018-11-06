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
"""ResNets, implemented in Gluon."""
from __future__ import division

from binary_models.model_parameters import ModelParameters

__all__ = ['ResNetExp', 'ResNetCParameters',
           'BasicBlockExp',
           'resnet18_c', 'resnet34_c', 'resnet50_c', 'resnet101_c', 'resnet152_c'
           ]

import os

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import base


# Helpers
def _conv3x3(bits, channels, stride, in_channels):
    return nn.QConv2D(channels, bits=bits, kernel_size=3,
                      strides=stride, padding=1, in_channels=in_channels)


class ScaledBinaryConv(HybridBlock):
    r"""ScaledBinaryConv implements scaled binarized 2D convolution,
        introduced by XNOR-Net Paper
    """

    def __init__(self, bits, bits_a, channels, kernel_size, stride, padding=0, in_channels=0, clip_threshold=1.0,
                 prefix=None, **kwargs):
        super(ScaledBinaryConv, self).__init__(**kwargs)
        self.qact = nn.QActivation(bits=bits_a, gradient_cancel_threshold=clip_threshold)
        self.qconv = nn.QConv2D(channels, bits=bits, kernel_size=kernel_size, strides=stride, padding=padding,
                                in_channels=in_channels, prefix=prefix, no_offset=True, apply_scaling=True)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels

    def hybrid_forward(self, F, x):
        y = self.qconv(self.qact(x))
        A = x.abs().mean(axis=1, keepdims=True)
        k = F.full((1, 1, self.kernel_size, self.kernel_size), 1 / self.kernel_size ** 2)
        K = F.Convolution(A, k, bias=None, name='scaling_conv', num_filter=1,
                          kernel=(self.kernel_size, self.kernel_size), no_bias=True, stride=(self.stride, self.stride),
                          pad=(self.padding, self.padding), layout='NCHW')
        K = F.stop_gradient(K)
        return F.broadcast_mul(K, y)


# Blocks
class BasicBlockExp(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, bits, bits_a, channels, stride, downsample=False, in_channels=0, clip_threshold=1.0, modifier=[],
                 write_on=-1, slices=-1, **kwargs):
        super(BasicBlockExp, self).__init__(**kwargs)
        self.bits = bits
        self.bits_a = bits_a
        self.channels = channels
        self.stride = stride
        self.in_channels = in_channels
        self.clip_threshold = clip_threshold
        self.write_on = write_on
        self.slices = slices

        self.act = None
        self.conv = None
        self.bn = None
        self.downsample = None
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
        self.modifier = modifier

        if 'scaled' in self.modifier:
            self._init_scaled()
        else:
            self._init_standard()

    def _init_standard(self):
        self.act = nn.QActivation(bits=self.bits_a, gradient_cancel_threshold=self.clip_threshold)
        self.conv = _conv3x3(self.bits, self.channels // self.slices, self.stride, self.in_channels)
        self.bn = nn.BatchNorm()

        if self.downsample is not None:
            self.downsample.add(nn.QActivation(bits=self.bits_a, gradient_cancel_threshold=self.clip_threshold))
            self.downsample.add(
                nn.QConv2D(self.channels, kernel_size=1, strides=self.stride, in_channels=self.in_channels,
                           prefix="sc_qconv_"))
            self.downsample.add(nn.BatchNorm())

    def _init_scaled(self):
        self.conv = ScaledBinaryConv(self.bits, self.bits_a, self.channels, 3, self.stride, padding=1,
                                     in_channels=self.in_channels, clip_threshold=self.clip_threshold)
        self.bn = nn.BatchNorm()

        if self.downsample is not None:
            self.downsample.add(ScaledBinaryConv(self.bits, self.bits_a, self.channels, 1, self.stride, padding=0,
                                                 in_channels=self.in_channels, clip_threshold=self.clip_threshold))
            self.downsample.add(nn.BatchNorm())

    def hybrid_forward(self, F, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        if self.act is not None:
            x = self.act(x)
        x = self.conv(x)
        x = self.bn(x)

        if self.slices == 1:
            return x + residual

        parts = []
        for index in range(self.slices):
            slice_width = self.channels // self.slices
            part = F.slice_axis(residual, axis=1, begin=slice_width*index, end=slice_width*(index + 1),)
            if index == self.write_on:
                part = part + x
            parts.append(part)
        return F.concat(*parts, dim=1)


# Nets
class ResNetExp(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False, bits=None, bits_a=None,
                 clip_threshold=1.0, slices=1, modifier=[], **kwargs):
        super(ResNetExp, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        assert bits is not None and bits_a is not None, "number of bits needs to be set"
        self.bits = bits
        self.bits_a = bits_a
        self.clip_threshold = clip_threshold
        self.slices = slices

        self.features = nn.HybridSequential(prefix='')
        self.features.add(nn.BatchNorm(scale=False, epsilon=2e-5))
        if thumbnail:
            self.features.add(nn.Conv2D(channels[0], kernel_size=3, strides=1, padding=1, in_channels=0,
                                        use_bias=False))
            # MXNet has a batch norm here, binary resnet performs better without
            # self.features.add(nn.BatchNorm())
        else:
            self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))
            self.features.add(nn.BatchNorm())

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                               stride, i+1, in_channels=channels[i], modifier=modifier))

        # v1 MXNet example has these deactivated, blocks finish with batchnorm and relu
        # but we need the relu, since we do not have activition in blocks
        # self.features.add(nn.BatchNorm())
        self.features.add(nn.Activation('relu'))

        self.features.add(nn.GlobalAvgPool2D())
        self.features.add(nn.Flatten())

        self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0, **kwargs):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        # more connections if we make a shortcut between each layer, but we need to double number of blocks
        layers = layers * 2
        # blocks will only add new features on a part of the branch
        layers = layers * self.slices
        with layer.name_scope():
            layer.add(block(self.bits, self.bits_a, channels, stride, channels != in_channels, in_channels=in_channels,
                            clip_threshold=self.clip_threshold, prefix='',
                            write_on=0, slices=self.slices, **kwargs))
            for i in range(layers-1):
                write_on = (1 + i) % self.slices
                layer.add(block(self.bits, self.bits_a, channels, 1, False, in_channels=channels,
                                clip_threshold=self.clip_threshold, prefix='',
                                write_on=write_on, slices=self.slices, **kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class ResNetCParameters(ModelParameters):
    def __init__(self):
        super(ResNetCParameters, self).__init__('ResNetC')

    def _is_it_this_model(self, opt):
        return opt.model.startswith('resnet') and opt.model.endswith("_c")

    def _map_opt_to_kwargs(self, opt, kwargs):
        assert opt.slices in [1, 2, 4, 8], ""
        kwargs['slices'] = opt.slices

    def _add_arguments(self, parser):
        parser.add_argument('--slices', type=int, default=4,
                            help='in how many slices should a block be split')


# Specification
resnet_spec = {18: ([2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ([3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ([3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ([3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ([3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_block_versions = [{'basic_block': BasicBlockExp, 'bottle_neck': BasicBlockExp}]


# Constructor
def get_resnet(num_layers, pretrained=False, ctx=cpu(),
               root=os.path.join(base.data_dir(), 'models'), **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    layers, channels = resnet_spec[num_layers]
    net = ResNetExp(BasicBlockExp, layers, channels, **kwargs)
    if pretrained:
        raise ValueError("No pretrained model exists, yet.")
        # from ..model_store import get_model_file
        # net.load_parameters(get_model_file('resnet%d_v%d'%(num_layers, version),
        #                                    root=root), ctx=ctx)
    return net

def resnet18_c(**kwargs):
    r"""ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(18, **kwargs)

def resnet34_c(**kwargs):
    r"""ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(34, **kwargs)

def resnet50_c(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(50, **kwargs)

def resnet101_c(**kwargs):
    r"""ResNet-101 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(101, **kwargs)

def resnet152_c(**kwargs):
    r"""ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_resnet(152, **kwargs)

