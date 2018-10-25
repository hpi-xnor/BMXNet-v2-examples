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
"""BiNets, implemented in Gluon."""
from __future__ import division

__all__ = ['Binet',
           'BasicBlockV1',
           'binet18', 'binet34',
           'get_binet']


import os

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import base



# Helpers
def _conv3x3(bits, channels, stride, in_channels):
    return nn.QConv2D(channels, bits=bits, kernel_size=3,
                      strides=stride, padding=1, in_channels=in_channels)


# Blocks
class BasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Bi-Real Net: Enhancing the Performance of 1-bit CNNs"
    <https://arxiv.org/abs/1808.00278>`_ paper.
    This is used for BiNet V1 for 18, 34 layers.

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

    def __init__(self, bits, bits_a, channels, stride, downsample=False, in_channels=0, clip_threshold=1.0, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.layer1 = nn.HybridSequential(prefix='')
        # Dif to Resnet: One layer is Sign + conv + batchnorm. There are shortcuts around all layers
        self.layer1.add(nn.QActivation(bits=bits_a, gradient_cancel_threshold=clip_threshold))
        self.layer1.add(_conv3x3(bits, channels, stride, in_channels))
        self.layer1.add(nn.BatchNorm())

        self.layer2 = nn.HybridSequential(prefix='')
        self.layer2.add(nn.QActivation(bits=bits_a, gradient_cancel_threshold=clip_threshold))
        self.layer2.add(_conv3x3(bits, channels, 1, channels))
        self.layer2.add(nn.BatchNorm())

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.AvgPool2D(pool_size=2, strides=2, padding=0))
            self.downsample.add(nn.QConv2D(channels, kernel_size=1, strides=1, in_channels=in_channels, prefix="sc_qconv_"))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):

        origin = x
        x = self.layer1(x)
        if self.downsample:
            origin = self.downsample(origin)
        residual = x + origin  # setting residual only to x instead seems to improve performance
        x = self.layer2(residual)

        return residual + x


# Nets
class Binet(HybridBlock):
    r""" From `"Bi-Real Net: Enhancing the Performance of 1-bit CNNs"
    <https://arxiv.org/abs/1808.00278>`_ paper.

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
        Enable thumbnail. Needed for small input images like cifar.
    """

    def __init__(self, block, layers, channels, classes=1000, thumbnail=True, bits=None, bits_a=None,
                 clip_threshold=1.0, modifier=[],  **kwargs):
        super(Binet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        assert bits is not None and bits_a is not None, "number of bits needs to be set"
        self.bits = bits
        self.bits_a = bits_a
        self.clip_threshold = clip_threshold

        self.features = nn.HybridSequential(prefix='')
        self.features.add(nn.BatchNorm(scale=False, epsilon=2e-5))
        if thumbnail:
            self.features.add(nn.Conv2D(channels[0], kernel_size=3, strides=1, padding=1, in_channels=0,
                                        use_bias=False))
        else:
            self.features.add(nn.Conv2D(channels[0], kernel_size=7, strides=2, padding=3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))

        for i, num_layer in enumerate(layers):
            stride = 1 if i == 0 else 2
            self.features.add(self._make_layer(block, num_layer, channels[i + 1],
                                               stride, i + 1, in_channels=channels[i]))

        self.features.add(nn.BatchNorm())
        self.features.add(nn.Activation('relu'))

        self.features.add(nn.GlobalAvgPool2D())
        self.features.add(nn.Flatten())

        self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(self.bits, self.bits_a, channels, stride, downsample=channels != in_channels, in_channels=in_channels,
                            clip_threshold=self.clip_threshold, prefix=''))
            for _ in range(layers - 1):
                layer.add(block(self.bits, self.bits_a, channels, 1, False, in_channels=channels,
                                clip_threshold=self.clip_threshold, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_block_versions = [{'basic_block': BasicBlockV1}]


# Constructor
def get_binet(num_layers, pretrained=False, ctx=cpu(),
               root=os.path.join(base.data_dir(), 'models'), **kwargs):
    r""" From `"Bi-Real Net: Enhancing the Performance of 1-bit CNNs"
    <https://arxiv.org/abs/1808.00278>`_ paper.

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
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    block_class = resnet_block_versions[0][block_type]
    net = Binet(block_class, layers, channels, **kwargs)
    if pretrained:
        raise ValueError("No pretrained model exists, yet.")
        # from ..model_store import get_model_file
        # net.load_parameters(get_model_file('resnet%d_v%d'%(num_layers, version),
        #                                    root=root), ctx=ctx)
    return net


def binet18(**kwargs):
    r""" From `"Bi-Real Net: Enhancing the Performance of 1-bit CNNs"
    <https://arxiv.org/abs/1808.00278>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_binet(18, **kwargs)


def binet34(**kwargs):
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
    return get_binet(34, **kwargs)
