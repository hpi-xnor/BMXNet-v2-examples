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

import argparse, time, os
import logging

import mxnet as mx
from mxnet import gluon
from mxnet import profiler
from mxnet.gluon.model_zoo import vision as models
import binary_models
from mxnet import autograd
from mxnet.test_utils import get_mnist_iterator
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
from contextlib import redirect_stdout
import numpy as np

from data import *


# CLI
def get_parser(evaluation=False):
    training = not evaluation
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    if training:
        parser.add_argument('--augmentation-level', type=int, choices=[1, 2, 3], default=1,
                            help='augmentation level, default is 1, possible values are: 1, 2, 3.')
        parser.add_argument('--batch-norm', action='store_true',
                            help='enable batch normalization or not in vgg. default is false.')
        parser.add_argument('--bits', type=int, default=32,
                            help='number of weight bits')
        parser.add_argument('--bits-a', type=int, default=32,
                            help='number of bits for activation')
        parser.add_argument('--clip-threshold', type=float, default=1.0,
                            help='clipping threshold, default is 1.0.')
        parser.add_argument('--epochs', type=int, default=120,
                            help='number of training epochs.')
        parser.add_argument('--gpus', type=str, default='',
                            help='ordinates of gpus to use, can be "0,1,2" or empty for cpu only.')
        parser.add_argument('--initialization', type=str, choices=["default", "gaussian"], default="default",
                            help='weight initialization, default is xavier with magnitude 2.')
        parser.add_argument('--kvstore', type=str, default='device',
                            help='kvstore to use for trainer/module.')
        parser.add_argument('--log', type=str, default='image-classification.log',
                            help='Filename and path where log file should be stored.')
        parser.add_argument('--log-interval', type=int, default=50,
                            help='Number of batches to wait before logging.')
        parser.add_argument('--lr', type=float, default=0.1,
                            help='learning rate. default is 0.1.')
        parser.add_argument('--lr-factor', default=0.1, type=float,
                            help='learning rate decay ratio')
        parser.add_argument('--lr-steps', default='30,60,90', type=str,
                            help='list of learning rate decay epochs as in str')
        parser.add_argument('--mode', type=str,
                            help='mode in which to train the model. options are symbolic, imperative, hybrid')
        parser.add_argument('--model', type=str, required=True,
                            help='type of model to use. see vision_model for options.')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum value for optimizer, default is 0.9.')
        parser.add_argument('--optimizer', type=str, default="sgd",
                            help='the optimizer to use. default is sgd.')
        parser.add_argument('--plot-network', type=str, default=None,
                            help='Whether to output the network plot.')
        parser.add_argument('--profile', action='store_true',
                            help='Option to turn on memory profiling for front-end, and prints out '
                                 'the memory usage by python function at the end.')
        parser.add_argument('--resume', type=str, default='',
                            help='path to saved weight where you want resume')
        parser.add_argument('--save-frequency', default=10, type=int,
                            help='epoch frequence to save model, best model will always be saved')
        parser.add_argument('--seed', type=int, default=123,
                            help='random seed to use. Default=123.')
        parser.add_argument('--start-epoch', default=0, type=int,
                            help='starting epoch, 0 for fresh training, > 0 to resume')
        parser.add_argument('--use-pretrained', action='store_true',
                            help='enable using pretrained model from gluon.')
        parser.add_argument('--wd', type=float, default=0.0001,
                            help='weight decay rate. default is 0.0001.')
        parser.add_argument('--write-summary', type=str, default=None,
                            help='write tensorboard summaries to this path')
    if evaluation:
        parser.add_argument('--gpu', type=str, default='',
                            help='ordinate of gpu to use, empty for cpu only.')
        parser.add_argument('--augmentation-level', type=int, default=0, help='dummy value')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--builtin-profiler', type=int, default=0,
                        help='Enable built-in profiler (0=off, 1=on)')
    parser.add_argument('--data-dir', type=str, default='',
                        help='training directory of imagenet images, contains train/val subdirs.')
    parser.add_argument('--data-path', type=str, default='.',
                        help='training directory where cifar10 / mnist data should be or is saved.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'imagenet', 'dummy'],
                        help='dataset to use. options are mnist, cifar10, imagenet and dummy.')
    parser.add_argument('--dtype', default='float32', type=str,
                        help='data type, float32 or float16 if applicable')
    parser.add_argument('--mean-subtraction', action="store_true",
                        help='whether to subtract ImageNet mean from data')
    parser.add_argument('--num-worker', '-j', dest='num_workers', default=4, type=int,
                        help='number of workers of dataloader.')
    parser.add_argument('--prefix', default='', type=str,
                        help='path to checkpoint prefix, default is current working dir')
    return parser


def get_model(model, ctx, opt):
    """Model initialization."""
    kwargs = {'ctx': ctx, 'pretrained': opt.use_pretrained, 'classes': classes}
    if model.startswith('resnet') and dataset == "cifar10":
        kwargs['thumbnail'] = True
    elif model.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm
    if model.startswith('resnet'):
        kwargs['clip_threshold'] = opt.clip_threshold

    net = binary_models.get_model(model, bits=opt.bits, bits_a=opt.bits_a, **kwargs)

    if opt.resume:
        net.load_parameters(opt.resume)
    elif not opt.use_pretrained:
        if model in ['alexnet']:
            net.initialize(mx.init.Normal(), ctx=ctx)
        else:
            net.initialize(get_initializer(), ctx=ctx)
    net.cast(opt.dtype)
    return net


def get_shape(dataset):
    if dataset == 'mnist':
        return (1, 1, 28, 28)
    elif dataset == 'cifar10':
        return (1, 3, 32, 32)
    elif dataset == 'imagenet' or dataset == 'dummy':
        return (1, 3, 299, 299) if model_name == 'inceptionv3' else (1, 3, 224, 224)


def get_initializer():
    if opt.initialization == "default":
        return mx.init.Xavier(magnitude=2)
    if opt.initialization == "gaussian":
        return mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)


def get_data_iters(opt, dataset, batch_size, num_workers=1, rank=0):
    """get dataset iterators"""
    if dataset == 'mnist':
        train_data, val_data = get_mnist_iterator(batch_size, (1, 28, 28),
                                                  num_parts=num_workers, part_index=rank)
    elif dataset == 'cifar10':
        train_data, val_data = get_cifar10_iterator(batch_size, (3, 32, 32), num_parts=num_workers, part_index=rank,
                                                    dir=opt.data_path, aug_level=opt.augmentation_level,
                                                    mean_subtraction=opt.mean_subtraction)
    elif dataset == 'imagenet':
        if not opt.data_dir:
            raise ValueError('Dir containing rec files is required for imagenet, please specify "--data-dir"')
        if model_name == 'inceptionv3':
            train_data, val_data = get_imagenet_iterator(opt.data_dir, batch_size, opt.num_workers, 299, opt.dtype)
        else:
            train_data, val_data = get_imagenet_iterator(opt.data_dir, batch_size, opt.num_workers, 224, opt.dtype)
    elif dataset == 'dummy':
        if model_name == 'inceptionv3':
            train_data, val_data = dummy_iterator(batch_size, (3, 299, 299))
        else:
            train_data, val_data = dummy_iterator(batch_size, (3, 224, 224))
    return train_data, val_data


def test(ctx, val_data):
    metric.reset()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()


def update_learning_rate(lr, trainer, epoch, ratio, steps):
    """Set the learning rate to the initial value decayed by ratio every N epochs."""
    new_lr = lr * (ratio ** int(np.sum(np.array(steps) < epoch)))
    trainer.set_learning_rate(new_lr)
    return trainer


def save_checkpoint(epoch, top1, best_acc):
    if opt.save_frequency and (epoch + 1) % opt.save_frequency == 0:
        fname = os.path.join(opt.prefix, '%s_%sbit_%04d_acc_%.4f.params' % (opt.model, opt.bits, epoch, top1))
        net.save_parameters(fname)
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, top1)
    if top1 > best_acc[0]:
        best_acc[0] = top1
        fname = os.path.join(opt.prefix, '%s_%sbit_best.params' % (opt.model, opt.bits))
        net.save_parameters(fname)
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, top1)


def get_dummy_data(data_iter, ctx):
    shapes = ((1,) + data_iter.provide_data[0].shape[1:], (1,) + data_iter.provide_label[0].shape[1:])
    return [mx.nd.array(np.zeros(shape), ctx=ctx) for shape in shapes]


def get_optimizer(opt):
    params = {'learning_rate': opt.lr, 'wd': opt.wd, 'multi_precision': True}
    if opt.optimizer == "sgd":
        params['momentum'] = opt.momentum
    return opt.optimizer, params


def train(opt, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    kv = mx.kv.create(opt.kvstore)
    train_data, val_data = get_data_iters(opt, dataset, batch_size, kv.num_workers, kv.rank)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), *get_optimizer(opt), kvstore = kv)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # dummy forward pass to initialize binary layers
    with autograd.record():
        data, label = get_dummy_data(train_data, ctx[0])
        output = net(data)

    summary_writer = None
    if opt.write_summary:
        from mxboard import SummaryWriter
        summary_writer = SummaryWriter(logdir=opt.write_summary, flush_secs=30)

    # set batch norm wd to zero
    params = net.collect_params('.*batchnorm.*')
    for key in params:
        params[key].wd_mult = 0.0

    if opt.plot_network is not None:
        x = mx.sym.var('data')
        sym = net(x)
        with open('{}.txt'.format(opt.plot_network), 'w') as f:
            with redirect_stdout(f):
                mx.viz.print_summary(sym, shape={"data": get_shape(dataset)})
        a = mx.viz.plot_network(sym, shape={"data": get_shape(dataset)})
        a.render('{}.gv'.format(opt.plot_network), view=True)

    global_step = 0
    if summary_writer:
        params = net.collect_params(".*weight|.*bias")
        for name, param in params.items():
            summary_writer.add_histogram(tag=name, values=param.data(ctx[0]),
                                         global_step=global_step, bins=1000)
            summary_writer.add_histogram(tag="%s-grad" % name, values=param.grad(ctx[0]),
                                         global_step=global_step, bins=1000)

    total_time = 0
    num_epochs = 0
    best_acc = [0]
    for epoch in range(opt.start_epoch, opt.epochs):
        trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor, lr_steps)
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0].astype(opt.dtype), ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with autograd.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                autograd.backward(Ls)
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if opt.log_interval and not (i+1) % opt.log_interval:
                name, acc = metric.get()
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f, %s=%f'%(
                    epoch, i, batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))
                if summary_writer:
                    summary_writer.add_scalar("batch-%s" % name[0], acc[0], global_step=global_step)
                    summary_writer.add_scalar("batch-%s" % name[1], acc[1], global_step=global_step)
            btic = time.time()
            global_step += 1

        epoch_time = time.time()-tic

        if summary_writer:
            params = net.collect_params(".*weight|.*bias")
            for name, param in params.items():
                summary_writer.add_histogram(tag=name, values=param.data(ctx[0]),
                                             global_step=global_step, bins=1000)
                summary_writer.add_histogram(tag="%s-grad" % name, values=param.grad(ctx[0]),
                                             global_step=global_step, bins=1000)

        # First epoch will usually be much slower than the subsequent epics,
        # so don't factor into the average
        if num_epochs > 0:
            total_time = total_time + epoch_time
        num_epochs = num_epochs + 1

        # train
        name, acc = metric.get()
        logger.info('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name[0], acc[0], name[1], acc[1]))
        logger.info('[Epoch %d] time cost: %f'%(epoch, epoch_time))
        if summary_writer:
            summary_writer.add_scalar("epoch", epoch, global_step=global_step)
            summary_writer.add_scalar("epoch-time", epoch_time, global_step=global_step)
            summary_writer.add_scalar("training-%s" % name[0], acc[0], global_step=global_step)
            summary_writer.add_scalar("training-%s" % name[1], acc[1], global_step=global_step)

        # test
        name, val_acc = test(ctx, val_data)
        logger.info('[Epoch %d] validation: %s=%f, %s=%f'%(epoch, name[0], val_acc[0], name[1], val_acc[1]))
        if summary_writer:
            summary_writer.add_scalar("validation-%s" % name[0], val_acc[0], global_step=global_step)
            summary_writer.add_scalar("validation-%s" % name[1], val_acc[1], global_step=global_step)

        # save model if meet requirements
        save_checkpoint(epoch, val_acc[0], best_acc)
    if num_epochs > 1:
        print('Average epoch time: {}'.format(float(total_time)/(num_epochs - 1)))

    if opt.mode != 'hybrid':
        net.hybridize()
        # dummy forward pass to save model
        with autograd.record():
            data, label = get_dummy_data(train_data, ctx[0])
            output = net(data)
    net.export(os.path.join(opt.prefix, "image-classifier-{}bit".format(opt.bits)), epoch=0)


def main():
    if opt.builtin_profiler > 0:
        profiler.set_config(profile_all=True, aggregate_stats=True)
        profiler.set_state('run')
    if opt.mode == 'symbolic':
        kv = mx.kv.create(opt.kvstore)
        train_data, val_data = get_data_iters(opt, dataset, batch_size, kv.num_workers, kv.rank)

        # dummy forward pass with gluon to initialize binary layers
        with autograd.record():
            data, label = get_dummy_data(train_data, context[0])
            output = net(data)

        data = mx.sym.var('data')
        out = net(data)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=context if num_gpus > 0 else [mx.cpu()])
        optimizer, optimizer_params = get_optimizer(opt)
        model_path = os.path.join(opt.prefix, 'image-classifier-%s' % opt.model)
        eval_metric = ['accuracy', mx.metric.create('top_k_accuracy', top_k=5)]
        mod.fit(train_data,
                eval_data = val_data,
                eval_metric = eval_metric,
                num_epoch=opt.epochs,
                kvstore=kv,
                batch_end_callback = mx.callback.Speedometer(batch_size, max(1, opt.log_interval)),
                epoch_end_callback = mx.callback.do_checkpoint(model_path),
                optimizer = optimizer,
                optimizer_params = optimizer_params,
                initializer = get_initializer())
        mod.save_params('%s-%d-final.params' % (model_path, opt.epochs))
    else:
        if opt.mode == 'hybrid':
            net.hybridize()
        train(opt, context)
    if opt.builtin_profiler > 0:
        profiler.set_state('stop')
        print(profiler.dumps())


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO)
    fh = logging.FileHandler(opt.log)
    logger = logging.getLogger()
    logger.addHandler(fh)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logging.debug('\n%s', '-' * 100)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)

    # global variables
    logger.info('Starting new image-classification task:, %s', opt)
    mx.random.seed(opt.seed)
    model_name = opt.model
    dataset_classes = {'mnist': 10, 'cifar10': 10, 'imagenet': 1000, 'dummy': 1000}
    batch_size, dataset, classes = opt.batch_size, opt.dataset, dataset_classes[opt.dataset]
    context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]
    num_gpus = len(context)
    batch_size *= max(1, num_gpus)
    lr_steps = [int(x) for x in opt.lr_steps.split(',') if x.strip()]
    metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])

    net = get_model(opt.model, context, opt)

    if opt.profile:
        import hotshot, hotshot.stats
        prof = hotshot.Profile('image-classifier-%s-%s.prof'%(opt.model, opt.mode))
        prof.runcall(main)
        prof.close()
        stats = hotshot.stats.load('image-classifier-%s-%s.prof'%(opt.model, opt.mode))
        stats.strip_dirs()
        stats.sort_stats('cumtime', 'calls')
        stats.print_stats()
    else:
        main()
