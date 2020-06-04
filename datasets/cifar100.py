from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms

from datasets.base import Dataset


class Cifar100(Dataset):
    name = "cifar100"
    num_classes = 100
    num_examples = 50000
    default_save_frequency = 10
    shape = (1, 3, 32, 32)

    def get_data(self, opt):
        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            return data, label

        transform_train = transforms.Compose([
            gcv_transforms.RandomCrop(32, pad=4),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(train=True, fine_label=True).transform_first(transform_train),
            batch_size=opt.batch_size, shuffle=True, last_batch='discard', num_workers=opt.num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(train=False, fine_label=True).transform_first(transform_test),
            batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        return train_data, val_data, batch_fn
