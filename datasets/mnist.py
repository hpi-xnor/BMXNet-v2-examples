import numpy as np
from mxnet import gluon

from .base import Dataset


class Mnist(Dataset):
    name = "mnist"
    num_classes = 10
    num_examples = 60000
    default_save_frequency = 10
    shape = (1, 3, 28, 28)

    def get_data(self, opt):
        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            return data, label

        def transformer(data, label):
            data = data.astype(np.float32) / 255
            return data, label

        train_data = gluon.data.DataLoader(
            gluon.data.vision.MNIST(train=True, transform=transformer),
            batch_size=opt.batch_size, shuffle=True, last_batch='discard')

        val_data = gluon.data.DataLoader(
            gluon.data.vision.MNIST(train=False, transform=transformer),
            batch_size=opt.batch_size, shuffle=False)
        return train_data, val_data, batch_fn
