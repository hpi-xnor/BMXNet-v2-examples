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

__all__ = ['ResNetE1', 'ResNetE2',
           'BasicBlockE1', 'BasicBlockE2',
           'resnet18_e1', 'resnet34_e1', 'resnet50_e1', 'resnet101_e1', 'resnet152_e1',
           'resnet18_e2', 'resnet34_e2', 'resnet50_e2', 'resnet101_e2', 'resnet152_e2',
           'ResNetEParameters'
           ]

import os

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import base

from .resnet import BasicBlockV1, BasicBlockV2


# Blocks
class BasicBlockE1(BasicBlockV1):
    r"""BasicBlock E1 similar to Bi-real network
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet E1 for 18, 34 layers.

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

    def __init__(self, *args, use_fp=False, use_pooling=False, write_on=0, slices=1, **kwargs):
        super(BasicBlockE1, self).__init__(*args, init=False, **kwargs)
        self.use_fp = use_fp
        self.use_pooling = use_pooling
        self.write_on = write_on
        self.slices = slices
        self.slice_width = self.channels // self.slices

        self._init()

    def _init(self):
        self.body.add(nn.activated_conv(self.slice_width, kernel_size=3, stride=self.stride,
                                        padding=1, in_channels=self.in_channels))
        self.body.add(nn.BatchNorm())
        if self.downsample is not None:
            conv_stride = self.stride
            if self.use_pooling:
                conv_stride = 1
                self.downsample.add(nn.AvgPool2D(pool_size=2, strides=2, padding=0))
            if self.use_fp:
                self.downsample.add(nn.Conv2D(self.channels, kernel_size=1, strides=conv_stride, use_bias=False,
                                              in_channels=self.in_channels, prefix="sc_conv_"))
            else:
                self.downsample.add(nn.activated_conv(self.channels, kernel_size=1, stride=conv_stride, padding=0,
                                                      in_channels=self.in_channels, prefix="sc_qconv_"))
            self.downsample.add(nn.BatchNorm())

    def hybrid_forward(self, F, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        x = self.body(x)
        if self.slices == 1:
            return x + residual

        parts_indices = [0, self.slice_width * self.write_on, self.slice_width * (self.write_on + 1), self.channels]
        parts_add_x = [False, True, False]
        parts = []
        for add_x, slice_begin, slice_end in zip(parts_add_x, parts_indices[:-1], parts_indices[1:]):
            if slice_end - slice_begin == 0:
                continue
            result = F.slice_axis(residual, axis=1, begin=slice_begin, end=slice_end)
            if add_x:
                result = result + x
            parts.append(result)
        return F.concat(*parts, dim=1)


# Blocks
class BasicBlockE2(BasicBlockV2):
    r"""BasicBlock E2 similar to Bi-real network
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet E2 for 18, 34 layers.

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

    def __init__(self, *args, use_fp=False, use_pooling=False, write_on=0, slices=1, **kwargs):
        super(BasicBlockE2, self).__init__(*args, init=False, **kwargs)
        self.use_fp = use_fp
        self.use_pooling = use_pooling
        self.write_on = write_on
        self.slices = slices
        self.slice_width = self.channels // self.slices

        self._init()

    def _init(self):
        self.body.add(nn.activated_conv(self.slice_width, kernel_size=3, stride=self.stride,
                                        padding=1, in_channels=self.in_channels))
        if self.downsample is not None:
            conv_stride = self.stride
            if self.use_pooling:
                conv_stride = 1
                self.downsample.add(nn.AvgPool2D(pool_size=2, strides=2, padding=0))
            if self.use_fp:
                self.downsample.add(nn.Conv2D(self.channels, kernel_size=1, strides=conv_stride, use_bias=False,
                                              in_channels=self.in_channels, prefix="sc_conv_"))
            else:
                self.downsample.add(nn.activated_conv(self.channels, kernel_size=1, stride=conv_stride, padding=0,
                                                      in_channels=self.in_channels, prefix="sc_qconv_"))

    def hybrid_forward(self, F, x):
        bn = self.bn(x)
        if self.downsample:
            residual = self.downsample(bn)
        else:
            residual = x
        x = self.body(bn)
        if self.slices == 1:
            return x + residual

        parts_indices = [0, self.slice_width * self.write_on, self.slice_width * (self.write_on + 1), self.channels]
        parts_add_x = [False, True, False]
        parts = []
        for add_x, slice_begin, slice_end in zip(parts_add_x, parts_indices[:-1], parts_indices[1:]):
            if slice_end - slice_begin == 0:
                continue
            result = F.slice_axis(residual, axis=1, begin=slice_begin, end=slice_end)
            if add_x:
                result = result + x
            parts.append(result)
        return F.concat(*parts, dim=1)


class ResNetE(HybridBlock):
    def __init__(self, channels, classes, use_fp, use_pooling, slices, **kwargs):
        super(ResNetE, self).__init__(**kwargs)
        self.features = nn.HybridSequential(prefix='')
        self.output = nn.Dense(classes, in_units=channels[-1])
        self.use_fp = use_fp
        self.use_pooling = use_pooling
        self.slices = slices

    r"""Helper methods which are equal for both resnets"""
    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0, **kwargs):
        kwargs["use_fp"] = self.use_fp
        kwargs["use_pooling"] = self.use_pooling

        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        # this tricks adds shortcut connections between original resnet blocks
        # we multiple number of blocks by 2, but add only one layer instead of two in each block
        layers = layers*2
        if stride == 1:
            # if we have no downsampling we can slice all blocks
            offset = 1
            slices = self.slices
            layers_without_stride = layers * self.slices - 1
        else:
            # with downsampling we do not slice the first block
            offset = 0
            slices = 1
            layers_without_stride = (layers - 1) * self.slices
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels, prefix='',
                            slices=slices, write_on=0, **kwargs))
            for i in range(layers_without_stride):
                write_on = (offset + i) % self.slices
                layer.add(block(channels, 1, False, in_channels=channels, prefix='',
                                slices=self.slices, write_on=write_on, **kwargs))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Nets
class ResNetE1(ResNetE):
    r"""ResNet E1 model from
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

    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 use_fp=False, use_pooling=False, slices=1, **kwargs):
        super(ResNetE1, self).__init__(channels, classes, use_fp, use_pooling, slices, **kwargs)
        assert len(layers) == len(channels) - 1

        with self.name_scope():
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
                self.features.add(
                    self._make_layer(block, num_layer, channels[i + 1], stride, i + 1, in_channels=channels[i]))

            # v1 MXNet example has these deactivated, blocks finish with batchnorm and relu
            # but we need the relu, since we do not have activition in blocks
            # self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())


class ResNetE2(ResNetE):
    r"""ResNet E2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

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

    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 use_fp=False, use_pooling=False, slices=1, **kwargs):
        super(ResNetE2, self).__init__(channels, classes, use_fp, use_pooling, slices, **kwargs)
        assert len(layers) == len(channels) - 1

        with self.name_scope():
            self.features.add(nn.BatchNorm(scale=False, epsilon=2e-5))
            if thumbnail:
                self.features.add(nn.Conv2D(channels[0], kernel_size=3, strides=1, padding=1, in_channels=0))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(
                    self._make_layer(block, num_layer, channels[i + 1], stride, i + 1, in_channels=in_channels))
                in_channels = channels[i + 1]

            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())


class ResNetEParameters(ModelParameters):
    def __init__(self):
        super(ResNetEParameters, self).__init__('ResNetE')

    def _is_it_this_model(self, model):
        return model.startswith('resnet') and "_e" in model

    def _map_opt_to_kwargs(self, opt, kwargs):
        assert opt.slices in [1, 2, 4, 8], ""
        kwargs['slices'] = opt.slices
        kwargs['use_fp'] = opt.fp_downsample_sc
        kwargs['use_pooling'] = opt.pool_downsample_sc

    def _add_arguments(self, parser):
        parser.add_argument('--fp-downsample-sc', action="store_true",
                            help='whether to use full precision for the 1x1 convolution at the downsample shortcut')
        parser.add_argument('--pool-downsample-sc', action="store_true",
                            help='whether to use average pooling instead of stride 2 at the downsample shortcut')
        parser.add_argument('--slices', type=int, default=1, choices=[1, 2, 4, 8],
                            help='in how many slices should a block be split (1, 2, 4, or 8)')


# Specification
resnet_spec = {18: ([2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ([3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ([3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ([3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ([3, 8, 36, 3], [64, 256, 512, 1024, 2048])}
resnet_net_versions = [(ResNetE1, BasicBlockE1),
                       (ResNetE2, BasicBlockE2)]


# Constructor
def get_resnet_e(version, num_layers, pretrained=False, ctx=cpu(),
               root=os.path.join(base.data_dir(), 'models'), **kwargs):
    r"""ResNet V1 model similar to Bi-real network
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
    assert version >= 1 and version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2." % version
    resnet_class, block_class = resnet_net_versions[version-1]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        raise ValueError("No pretrained model exists, yet.")
        # from ..model_store import get_model_file
        # net.load_parameters(get_model_file('resnet%d_v%d'%(num_layers, version),
        #                                    root=root), ctx=ctx)
    return net

def resnet18_e1(**kwargs):
    r"""ResNet-18 V1 model similar to Bi-real network
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
    return get_resnet_e(1, 18, **kwargs)

def resnet34_e1(**kwargs):
    r"""ResNet-34 V1 model similar to Bi-real network
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
    return get_resnet_e(1, 34, **kwargs)

def resnet50_e1(**kwargs):
    r"""ResNet-50 V1 model similar to Bi-real network
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
    return get_resnet_e(1, 50, **kwargs)

def resnet101_e1(**kwargs):
    r"""ResNet-101 V1 model similar to Bi-real network
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
    return get_resnet_e(1, 101, **kwargs)

def resnet152_e1(**kwargs):
    r"""ResNet-152 V1 model similar to Bi-real network
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
    return get_resnet_e(1, 152, **kwargs)

def resnet18_e2(**kwargs):
    r"""ResNet-18 V2 model similar to Bi-real network
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
    return get_resnet_e(2, 18, **kwargs)

def resnet34_e2(**kwargs):
    r"""ResNet-34 V2 model similar to Bi-real network
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
    return get_resnet_e(2, 34, **kwargs)

def resnet50_e2(**kwargs):
    r"""ResNet-50 V2 model similar to Bi-real network
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
    return get_resnet_e(2, 50, **kwargs)

def resnet101_e2(**kwargs):
    r"""ResNet-101 V2 model similar to Bi-real network
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
    return get_resnet_e(2, 101, **kwargs)

def resnet152_e2(**kwargs):
    r"""ResNet-152 V2 model similar to Bi-real network
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
    return get_resnet_e(2, 152, **kwargs)
