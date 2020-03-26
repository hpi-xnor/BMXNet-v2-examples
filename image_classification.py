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

import os
import logging
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import mxnet as mx
from gluoncv.utils import LRScheduler, LRSequential
from graphviz import ExecutableNotFound
from mxnet import autograd
from mxnet import gluon
from mxnet import profiler
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric

import binary_models
from datasets.util import *
from util.arg_parser import get_parser
from util.log_progress import log_progress


# CLI: see util/arg_parser.py


def get_model_path(opt):
    return os.path.join(opt.prefix, 'image-classifier-%s' % opt.model)


def csv_args_dict(value):
    if value == "":
        return {}
    return {int(k): float(v) for k, v in [pair.split(":") for pair in value.split(",")]}


def _load_model(opt):
    model_prefix = get_model_path(opt)
    logger.info('Loaded model %s-%04d.params', model_prefix, opt.start_epoch)
    return mx.model.load_checkpoint(model_prefix, opt.start_epoch)


def get_model(opt, ctx):
    """Model initialization."""
    kwargs = {'ctx': ctx, 'pretrained': opt.use_pretrained, 'classes': get_num_classes(opt.dataset)}
    if opt.model.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm

    thumbnail_models = ['resnet', 'binet', 'densenet', 'meliusnet']
    if any(opt.model.startswith(name) for name in thumbnail_models) and get_shape(opt)[2] < 50:
        kwargs['initial_layers'] = "thumbnail"
    else:
        kwargs['initial_layers'] = opt.initial_layers

    for model_parameter in binary_models.get_model_parameters():
        model_parameter.set_args_for_model(opt, kwargs)

    skip_init = False
    arg_params, aux_params = None, None
    if opt.start_epoch > 0 and opt.mode == 'symbolic':
        net, arg_params, aux_params = _load_model(opt)
        skip_init = True
    else:
        with gluon.nn.set_binary_layer_config(bits=opt.bits, bits_a=opt.bits_a, approximation=opt.approximation,
                                              grad_cancel=opt.clip_threshold, activation=opt.activation_method,
                                              weight_quantization=opt.weight_quantization):
            net = binary_models.get_model(opt.model, **kwargs)

    if opt.resume:
        net.load_parameters(opt.resume)
    elif not opt.use_pretrained and not skip_init:
        if opt.model in ['alexnet']:
            net.initialize(mx.init.Normal(), ctx=ctx)
        else:
            net.initialize(get_initializer(), ctx=ctx)
    if opt.mode != 'symbolic':
        net.cast(opt.dtype)
    return net, arg_params, aux_params


def get_initializer():
    if opt.initialization == "default":
        return mx.init.Xavier(magnitude=2)
    if opt.initialization == "gaussian":
        return mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)
    if opt.initialization == "msraprelu_avg":
        return mx.init.MSRAPrelu()
    if opt.initialization == "msraprelu_in":
        return mx.init.MSRAPrelu(factor_type="in")


def test(ctx, val_data, batch_fn, testing=False):
    metric.reset()
    if hasattr(val_data, "reset"):
        val_data.reset()
    for batch in val_data:
        data, label = batch_fn(batch, ctx)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
        if testing:
            break
    return metric.get()


class LRTracker:
    def __init__(self, trainer, summary_writer):
        self.trainer = trainer
        self.prev_lr = trainer.learning_rate
        self.summary_writer = summary_writer

    def __call__(self, epoch, global_step=0):
        current_lr = self.trainer.learning_rate
        if current_lr != self.prev_lr:
            logger.info('[Epoch %d] Change learning rate to %f', epoch, current_lr)
            self.prev_lr = current_lr
        if self.summary_writer is not None:
            self.summary_writer.add_scalar("training/lr", current_lr, global_step=global_step)


def save_checkpoint(trainer, epoch, top1, best_acc, force_save=False):
    if opt.save_frequency and (epoch + 1) % opt.save_frequency == 0 or force_save:
        fname = os.path.join(opt.prefix, '%s_%sbit_%04d_acc_%.4f.{}' % (opt.model, opt.bits, epoch, top1))
        net.save_parameters(fname.format("params"))
        trainer.save_states(fname.format("states"))
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f',
                    epoch, fname.format("{params,states}"), top1)
    if top1 > best_acc:
        fname = os.path.join(opt.prefix, '%s_%sbit_best.{}' % (opt.model, opt.bits))
        net.save_parameters(fname.format("params"))
        trainer.save_states(fname.format("states"))
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f',
                    epoch, fname.format("{params,states}"), top1)


def get_dummy_data(opt, ctx):
    data_shape = get_shape(opt)
    shapes = ((1,) + data_shape[1:], (1,))
    return [mx.nd.array(np.zeros(shape), ctx=ctx) for shape in shapes]


def _get_lr_scheduler(opt):
    lr_factor = opt.lr_factor
    lr_steps = [int(i) for i in opt.lr_steps.split(',')]
    lr_steps = [e - opt.warmup_epochs for e in lr_steps]
    num_batches = get_num_examples(opt.dataset) // batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                    nepochs=opt.epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_steps,
                    step_factor=lr_factor, power=2)
    ])
    return lr_scheduler


def get_optimizer(opt):
    params = {
        'wd': opt.wd,
        'lr_scheduler': _get_lr_scheduler(opt)
    }
    if opt.dtype != 'float32':
        params['multi_precision'] = True
    if opt.optimizer == "sgd" or opt.optimizer == "nag":
        params['momentum'] = opt.momentum
    return opt.optimizer, params


def get_blocks(net, search_for_type, result=()):
    """
    Returns a tuple containing all layer objects of type search_for_type in net
    """
    for _, child in net._children.items():
        if isinstance(child, search_for_type):
            result = result + (child,)
        else:
            result = get_blocks(child, search_for_type, result=result)
    return result


def plot_network():
    x = mx.sym.var('data')
    sym = net(x)
    with open('{}.txt'.format(opt.plot_network), 'w') as f:
        with redirect_stdout(f):
            mx.viz.print_summary(sym, shape={"data": get_shape(opt)}, quantized_bitwidth=opt.bits)
    graph = mx.viz.plot_network(sym, shape={"data": get_shape(opt)})
    try:
        graph.render('{}.gv'.format(opt.plot_network))
    except OSError as e:
        logger.error(e)
    except ExecutableNotFound as e:
        logger.error(e)


def log_metrics(phase, name, acc, epoch, summary_writer, global_step, sep=": "):
    logger.info('[Epoch %d] %s%s%s=%f, %s=%f' % (epoch, phase, sep, name[0], acc[0], name[1], acc[1]))
    if summary_writer:
        summary_writer.add_scalar("%s/%s" % (name[0], phase), acc[0], global_step=global_step)
        summary_writer.add_scalar("%s/%s" % (name[1], phase), acc[1], global_step=global_step)


def write_net_summaries(summary_writer, single_ctx, global_step=0, write_grads=True):
    if summary_writer is None:
        return

    params = net.collect_params(".*weight|.*bias")
    for name, param in params.items():
        summary_writer.add_histogram(tag=name, values=param.data(single_ctx),
                                     global_step=global_step, bins=1000)
        if write_grads:
            summary_writer.add_histogram(tag="%s-grad" % name, values=param.grad(single_ctx),
                                         global_step=global_step, bins=1000)


def train(opt, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    kv = mx.kv.create(opt.kvstore)
    train_data, val_data, batch_fn = get_data_iters(opt)
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), *get_optimizer(opt), kvstore=kv)
    if opt.resume_states != '':
        trainer.load_states(opt.resume_states)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # dummy forward pass to initialize binary layers
    data, _ = get_dummy_data(opt, ctx[0])
    _ = net(data)

    if opt.mode == 'hybrid':
        net.hybridize()

    # set batch norm wd to zero
    params = net.collect_params('.*batchnorm.*')
    for key in params:
        params[key].wd_mult = 0.0

    if opt.plot_network is not None:
        plot_network()

    if opt.dry_run:
        return

    summary_writer = None
    if opt.write_summary:
        from mxboard import SummaryWriter
        summary_writer = SummaryWriter(logdir=opt.write_summary, flush_secs=60)
    write_net_summaries(summary_writer, ctx[0], write_grads=False)
    track_lr = LRTracker(trainer, summary_writer)

    total_time = 0
    num_epochs = 0
    best_acc = 0
    epoch_time = -1
    num_examples = get_num_examples(opt.dataset)

    for epoch in range(opt.start_epoch, opt.epochs):
        global_step = epoch * num_examples
        track_lr(epoch, global_step)
        tic = time.time()
        if hasattr(train_data, "reset"):
            train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)
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
            trainer.step(batch_size)
            metric.update(label, outputs)

            if opt.log_interval and not (i+1) % opt.log_interval:
                name, acc = metric.get()
                log_metrics("batch", name, acc, epoch, summary_writer, global_step,
                            sep=" [%d]\tSpeed: %f samples/sec\t" % (i, batch_size/(time.time()-btic)))
                log_progress(num_examples, opt, epoch, i, time.time()-tic, epoch_time)
                track_lr(epoch, global_step)

            btic = time.time()
            global_step += batch_size
            if opt.test_run:
                break

        epoch_time = time.time()-tic

        write_net_summaries(summary_writer, ctx[0], global_step=global_step)

        # First epoch will usually be much slower than the subsequent epics,
        # so don't factor into the average
        if num_epochs > 0:
            total_time = total_time + epoch_time
        num_epochs = num_epochs + 1

        logger.info('[Epoch %d] time cost: %f' % (epoch, epoch_time))
        if summary_writer:
            summary_writer.add_scalar("training/epoch", epoch, global_step=global_step)
            summary_writer.add_scalar("training/epoch-time", epoch_time, global_step=global_step)

        # train
        name, acc = metric.get()
        log_metrics("training", name, acc, epoch, summary_writer, global_step)

        # test
        name, val_acc = test(ctx, val_data, batch_fn, opt.test_run)
        log_metrics("validation", name, val_acc, epoch, summary_writer, global_step)

        if opt.interrupt_at is not None and epoch + 1 == opt.interrupt_at:
            logging.info("[Epoch %d] Interrupting run now because 'interrupt-at' was set to %d..." %
                         (epoch, opt.interrupt_at))
            save_checkpoint(trainer, epoch, val_acc[0], best_acc, force_save=True)
            sys.exit(3)

        # save model if meet requirements
        save_checkpoint(trainer, epoch, val_acc[0], best_acc)
        best_acc = max(best_acc, val_acc[0])

    if num_epochs > 1:
        print('Average epoch time: {}'.format(float(total_time)/(num_epochs - 1)))

    if opt.mode != 'hybrid':
        net.hybridize()
    # dummy forward pass to save model
    data, _ = get_dummy_data(opt, ctx[0])
    _ = net(data)
    net.export(os.path.join(opt.prefix, "image-classifier-{}bit".format(opt.bits)), epoch=0)


def train_symbolic(opt, ctx):
    kv = mx.kv.create(opt.kvstore)
    train_data, val_data, _ = get_data_iters(opt)

    if not opt.start_epoch > 0:
        if opt.plot_network is not None:
            plot_network()
    else:
        mod = mx.mod.Module(context=ctx, symbol=net)

    optimizer, optimizer_params = get_optimizer(opt)
    model_path = get_model_path(opt)
    eval_metric = ['accuracy', mx.metric.create('top_k_accuracy', top_k=5)]

    if opt.dry_run:
        return

    summary_writer = None
    if opt.write_summary:
        from mxboard import SummaryWriter
        summary_writer = SummaryWriter(logdir=opt.write_summary, flush_secs=60)

    batch_end_cbs = [
        mx.callback.Speedometer(batch_size, max(1, opt.log_interval))
    ]
    epoch_end_cbs = [
        mx.callback.do_checkpoint(model_path, period=opt.save_frequency)
    ]

    if summary_writer:
        def metric_callback(param):
            if not param.eval_metric or param.nbatch % opt.log_interval != 0:
                return
            for name, value in param.eval_metric.get_name_value():
                summary_writer.add_scalar(tag=name, value=value, global_step=param.epoch)
        batch_end_cbs.append(metric_callback)

        def param_callback(epoch, _, arg_params, __):
            for name in arg_params:
                summary_writer.add_histogram(tag=name, values=arg_params[name], global_step=epoch, bins=1000)
        epoch_end_cbs.append(param_callback)

    mod.fit(train_data,
            begin_epoch=opt.start_epoch,
            eval_data=val_data,
            eval_metric=eval_metric,
            num_epoch=opt.epochs,
            kvstore=kv,
            batch_end_callback=batch_end_cbs,
            epoch_end_callback=epoch_end_cbs,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            arg_params=arg_params,
            aux_params=aux_params,
            initializer=get_initializer())
    mod.save_params('%s-%d-final.params' % (model_path, opt.epochs))


def main():
    if opt.builtin_profiler > 0:
        profiler.set_config(profile_all=True, aggregate_stats=True)
        profiler.set_state('run')
    if opt.mode == 'symbolic':
        train_symbolic(opt, context)
    else:
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
    if opt.save_frequency is None:
        opt.save_frequency = get_default_save_frequency(opt.dataset)
    logger.info('Starting new image-classification task:, %s', opt)
    mx.random.seed(opt.seed)
    batch_size, dataset, classes = opt.batch_size, opt.dataset, get_num_classes(opt.dataset)
    context = [mx.gpu(int(i)) for i in opt.gpus.split(',')] if opt.gpus.strip() else [mx.cpu()]
    if opt.dry_run:
        context = [mx.cpu()]
    num_gpus = len(context)
    batch_size *= max(1, num_gpus)
    opt.batch_size = batch_size
    metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])

    net, arg_params, aux_params = get_model(opt, context)

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
