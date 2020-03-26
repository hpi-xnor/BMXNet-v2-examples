import math
import os
import mxnet as mx
import logging

from .base import Dataset

MEAN_RGB = [123.68, 116.779, 103.939]
STD_RGB = [58.393, 57.12, 57.375]


def as_dictionary(**kwargs):
    return kwargs


def log_dictionary(name, dictionary):
    sorted_keys = sorted(dictionary.keys())
    dict_str = ", ".join("'{}': {}".format(k, dictionary[k]) for k in sorted_keys)
    logging.info("RecordIterArgs for {}: {{{}}}".format(name, dict_str))
    return dictionary


def log_kwargs(name, **kwargs):
    return log_dictionary(name, kwargs)


def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers, input_size,
                 augmentation_level="low"):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)

    jitter_param = 0.4
    lighting_param = 0.1

    def batch_fn(batch, ctx):
        data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    common_args = as_dictionary(
        batch_size=batch_size,
        data_shape=(3, input_size, input_size),
        mean_r=MEAN_RGB[0],
        mean_g=MEAN_RGB[1],
        mean_b=MEAN_RGB[2],
        std_r=STD_RGB[0],
        std_g=STD_RGB[1],
        std_b=STD_RGB[2],
        preprocess_threads=num_workers,
    )

    train_args = as_dictionary(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        shuffle=True,
    )
    val_args = as_dictionary(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        shuffle=False,
    )

    train_args.update(common_args)
    val_args.update(common_args)

    if augmentation_level == "low":
        train_args.update(as_dictionary(
            rand_mirror=True,
            rand_crop=True,
        ))
    elif augmentation_level == "medium":
        train_args.update(as_dictionary(
            rand_mirror=True,
            random_resized_crop=True,
            max_aspect_ratio=4. / 3.,
            min_aspect_ratio=3. / 4.,
            max_random_area=1,
            min_random_area=0.08,
        ))
        crop_ratio = 0.875
        resize = int(math.ceil(input_size / crop_ratio))
        val_args.update(as_dictionary(
            resize=resize,
        ))
    elif augmentation_level == "high":
        train_args.update(as_dictionary(
            rand_mirror=True,
            random_resized_crop=True,
            max_aspect_ratio=4. / 3.,
            min_aspect_ratio=3. / 4.,
            max_random_area=1,
            min_random_area=0.08,
            brightness=jitter_param,
            saturation=jitter_param,
            contrast=jitter_param,
            pca_noise=lighting_param,
        ))
        crop_ratio = 0.875
        resize = int(math.ceil(input_size / crop_ratio))
        val_args.update(as_dictionary(
            resize=resize,
        ))
    train_data = mx.io.ImageRecordIter(**log_dictionary("train", train_args))
    val_data = mx.io.ImageRecordIter(**log_dictionary("val", val_args))
    return train_data, val_data, batch_fn


class Imagenet(Dataset):
    name = "imagenet"
    num_classes = 1000
    num_examples = 1281167
    shape = 1, 3, 224, 224

    def get_shape(self, opt):
        if opt.model == 'inceptionv3':
            return 1, 3, 299, 299
        return self.shape

    def get_data(self, opt):
        if not opt.data_dir:
            raise ValueError('Dir containing rec files is required for imagenet, please specify "--data-dir"')
        files_in_data_dir = os.listdir(opt.data_dir)
        if all(f in files_in_data_dir for f in ("train.rec", "train.idx", "val.rec", "val.idx")):
            rec_train = os.path.join(opt.data_dir, "train.rec")
            rec_train_idx = os.path.join(opt.data_dir, "train.idx")
            rec_val = os.path.join(opt.data_dir, "val.rec")
            rec_val_idx = os.path.join(opt.data_dir, "val.idx")
        else:
            rec_train = os.path.join(opt.data_dir, "imagenet1k-train.rec")
            rec_train_idx = os.path.join(opt.data_dir, "imagenet1k-train.idx")
            rec_val = os.path.join(opt.data_dir, "imagenet1k-val.rec")
            rec_val_idx = os.path.join(opt.data_dir, "imagenet1k-val.idx")
        return get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx,
                            opt.batch_size, opt.num_workers, self.get_shape(opt)[-1],
                            augmentation_level=opt.augmentation)
