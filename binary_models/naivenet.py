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
"""DenseNet, implemented in Gluon."""
from mxnet.gluon import HybridBlock, nn

from binary_models.basenet_dense import *
from binary_models.meliusnet import ImprovementBlock

__all__ = ['NaiveNet', 'NaiveNetParameters',
           'naivenet_flex', 'naivenet17']


class NaiveNet(BaseNetDense):
    def _add_base_block_structure(self, dilation):
        self._add_dense_block(dilation)
        self.current_stage.add(
            ImprovementBlock(self.num_features, self.num_features, dilation=dilation, prefix='')
        )


class NaiveNetParameters(BaseNetDenseParameters):
    def __init__(self):
        super(NaiveNetParameters, self).__init__('NaiveNet')

    def _is_it_this_model(self, model):
        return model.startswith('naivenet')


# Specification
naivenet_spec = {
    # name: block_config,     reduction_factors,                  downsampling
    None:   (None,            [1 / 2,     1 / 2,     1 / 2],      DOWNSAMPLE_STRUCT),
    '17':   ([3, 3, 3, 3],    [128 / 256, 160 / 320, 192 / 352],  DOWNSAMPLE_STRUCT),
}


# Constructor
get_naivenet = get_basenet_constructor(naivenet_spec, NaiveNet)


def naivenet_flex(**kwargs):
    return get_naivenet(None, **kwargs)


def naivenet17(**kwargs):
    return get_naivenet('17', **kwargs)
