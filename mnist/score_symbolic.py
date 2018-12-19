import numpy as np
import os
import urllib
import gzip
import struct
import argparse
import matplotlib.pyplot as plt
import mxnet as mx
import logging

from random import randint
from math import sqrt

def download_data(dir, files):
    for url in files:
        name = url.rsplit('/', 1)[-1]
        filename = os.path.join(dir, name)

        if not os.path.isfile(filename):
            logging.info("downloading file %s..." % name)
            urllib.urlretrieve(url, filename)

def read_data(label_url, image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_url, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

def prepare_data():
    path = 'data/'
    files = ['http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
             'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
             'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
             'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz']

    if not os.path.exists(path):
        os.makedirs(path)

    download_data(path, files)

    (train_lbl, train_img) = read_data(
        path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
        path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')
    return train_img, val_img, train_lbl, val_lbl

def check_data_visually(train_img, train_lbl):
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.imshow(train_img[i], cmap='Greys_r')
        plt.axis('off')
    plt.show()
    logging.info('label: %s' % (train_lbl[0:10],))


def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def prepair_data_iter(train_img, val_img, train_lbl, val_lbl, batch_size):   
    #train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)
    return val_iter


def val(model_prefix, epoch_num, train_img, val_img, train_lbl, val_lbl, batch_size, gpu_id=0):
    '''
    validate the model using mnist validation set.
    '''
    device = mx.cpu()
    if gpu_id >= 0:
        device = mx.gpu(gpu_id)
    
    logging.info('Preparing data for validation...')
    val_iter = prepair_data_iter(train_img, val_img, train_lbl, val_lbl, batch_size)
    
    logging.info('Loading model...')   
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch_num)
        
    #Note: we have to do this because the converted model from gluon without this argument.
    if not 'softmax_label' in sym.list_arguments():
        logging.info(sym.list_arguments())
        sym = mx.symbol.SoftmaxOutput(data=sym, name='softmax')
        logging.info(sym.list_arguments())

    model = mx.mod.Module(symbol=sym, context=device)
   
    logging.info('Model binding...')   
    model.bind(for_training=False,
             data_shapes=val_iter.provide_data,
             label_shapes=val_iter.provide_label)
    model.set_params(arg_params, aux_params)

    
    logging.info('Evaluating...')
    metric = mx.metric.Accuracy()
    score = model.score(val_iter, metric)
    logging.info (score)

def classify(val_img, model_prefix, epoch_num, train_img, train_lbl, val_lbl, batch_size, gpu_id=0):
    '''
    predict a single sample that randomly picked from the validation set
    '''
    device = mx.cpu()
    if gpu_id >= 0:
        device = mx.gpu(gpu_id)
    val_iter = prepair_data_iter(train_img, val_img, train_lbl, val_lbl, batch_size)

    logging.info('Loading model...')   
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch_num)
        
    #Note: we have to do this because the converted model from gluon without this argument.
    if not 'softmax_label' in sym.list_arguments():
        logging.info(sym.list_arguments())
        sym = mx.symbol.SoftmaxOutput(data=sym, name='softmax')
        logging.info(sym.list_arguments())

    model = mx.mod.Module(symbol=sym, context=device)
   
    logging.info('Model binding...')   
    model.bind(for_training=False,
             data_shapes=val_iter.provide_data,
             label_shapes=val_iter.provide_label)
    model.set_params(arg_params, aux_params)

    n = randint(0,batch_size-1)
    plt.imshow(val_img[n], cmap='Greys_r')
    plt.axis('off')
    plt.show()
    prob = model.predict(eval_data=val_iter, num_batch=1)[n].asnumpy() 
    logging.info ('Classified as %d[%d] with probability %f' % (prob.argmax(), val_lbl[n], max(prob)))


def check_data_visually(train_img, train_lbl):
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.imshow(train_img[i], cmap='Greys_r')
        plt.axis('off') 
    logging.info('label: %s' % (train_lbl[0:10],))
    plt.show()


def main(args):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    logging.info('preparing data...')
    train_img, val_img, train_lbl, val_lbl = prepare_data()
    #can be used for checking mnist data with respect to its label
    #check_data_visually(train_img, train_lbl)
    if not args.predict:
        val(args.model_prefix, args.epochs, train_img, val_img, train_lbl, val_lbl, args.bs, args.gpu_id)
    else:       
        classify(val_img, args.model_prefix, args.epochs, train_img, train_lbl, val_lbl, args.bs, args.gpu_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scoring script for mnist models')
    parser.add_argument('--model-prefix', dest='model_prefix', type=str, help="gives where to find the .param file and .json file")
    parser.add_argument('--gpus', dest='gpu_id', type=int, default=-1, help='selected gpu device id, otherwise cpu will be used')
    parser.add_argument('--epochs', dest='epochs', type=int, default=0, help='set the epoch number')
    parser.add_argument('--predict', dest='predict', action='store_true',default=False, help='will predict one sample randomly picked and show the visualisation result')
    parser.add_argument('--batch-size', dest='bs', type=int, default=100, help='batch size')
    args = parser.parse_args()
    main(args)