from datasets.cifar10 import Cifar10
from datasets.dummy_imagenet import DummyImagenet
from datasets.mnist import Mnist
from .base import Dataset
from .imagenet import Imagenet

__all__ = ["get_num_classes", "get_default_save_frequency", "get_num_examples", "get_shape", "get_data_iters",
           "get_all_dataset_names", "Dataset"]


ALL_DATASETS = [Imagenet, Cifar10, Mnist, DummyImagenet]


def _get_all_datasets():
    return [x() for x in ALL_DATASETS]


def get_all_dataset_names():
    return [x.name for x in _get_all_datasets()]


def _get_dataset_by_name(dataset_name):
    for dataset in _get_all_datasets():
        if dataset.name == dataset_name:
            return dataset
    raise RuntimeError("Dataset {} not found.".format(dataset_name))


def get_num_classes(dataset):
    return _get_dataset_by_name(dataset).num_classes


def get_default_save_frequency(dataset):
    return _get_dataset_by_name(dataset).default_save_frequency


def get_num_examples(dataset):
    return _get_dataset_by_name(dataset).num_examples


def get_shape(opt):
    return _get_dataset_by_name(opt.dataset).get_shape(opt)


def get_data_iters(opt):
    return _get_dataset_by_name(opt.dataset).get_data(opt)
