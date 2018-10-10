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

from __future__ import division

from functools import reduce
from operator import mul

import math
import numpy as np

import tqdm as tqdm
from mxnet import gluon

from datasets.data import *

from image_classification import get_parser, get_data_iters


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


parser = get_parser(evaluation=True)
opt = parser.parse_args()

json_file = 'model/deploy.json'
param_file = 'model/image-classifier-resnet18_v1-40-final.params'
ctx = mx.gpu(int(opt.gpu)) if opt.gpu.strip() else mx.cpu()

net = gluon.nn.SymbolBlock.imports(json_file, ['data'], param_file=param_file, ctx=ctx)

fp_weights = 0
binary_weights = 0

for param_name in net.params.keys():
    param = net.params[param_name]
    param_data = param.data()
    num_params = reduce(mul, param_data.shape, 1)
    if "stage" in param_name and "weight" in param_name:
        # print("binary")
        signed = param_data.det_sign()
        param.set_data(signed)
        binary_weights += num_params
    else:
        # print("full-precision")
        fp_weights += num_params
    # print(param_name)
    # print(param_data)
    # print("=" * 50)
bits_required = binary_weights + fp_weights * 32
bytes_required = bits_required / 8
print("full-precision weights: {}".format(fp_weights))
print("binary weights: {} ({:.2f}% of weights are binary)".format(
    binary_weights, 100 * binary_weights / (fp_weights + binary_weights))
)
print("compressed model size : ~{} ({:.2f}% binary)".format(
    convert_size(bytes_required), 100 * binary_weights / bits_required)
)

# from PIL import ImageFont
# font = ImageFont.truetype("/usr/share/fonts/truetype/hack/Hack-Regular.ttf", size=20)

num_correct = 0
num_wrong = 0

_, val_data = get_data_iters(opt)

for i, batch in enumerate(tqdm.tqdm(val_data)):
    if opt.gpu.strip():
        data = mx.nd.array(batch.data[0], ctx=ctx)
    else:
        data = mx.nd.array(batch.data[0], ctx=ctx)
    result = net(data)
    probabilities = result.softmax().asnumpy()
    ground_truth = batch.label[0].asnumpy()

    predictions = np.argmax(probabilities, axis=1)
    likeliness = np.max(probabilities, axis=1)

    num_correct += np.sum(predictions == ground_truth)
    num_wrong += np.sum(predictions != ground_truth)

    # from datasets.imagenet_classes import CLASSES
    # for i in range(opt.batch_size):
    #     transformed = image.asnumpy().astype(np.uint8).transpose(1, 2, 0)
    #     image = Image.fromarray(transformed, "RGB")
    #     draw = ImageDraw.ImageDraw(image)
    #     draw.text((0, 200), CLASSES[prediction] + " ({:.0f}%)".format(likeliness * 100), font=font,
    #               fill="green" if correct else "red")
    #
    #     image.show()
    #     print(CLASSES[np.argmax(probabilities)], np.max(probabilities))

print("Accuracy: {:.2f}%".format(100 * num_correct / (num_correct + num_wrong)))
