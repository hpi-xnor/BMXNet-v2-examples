import mxnet as mx

from datasets.imagenet import Imagenet


class DummyIter(mx.io.DataIter):
    def __init__(self, batch_size, data_shape, batches=100):
        super(DummyIter, self).__init__(batch_size)
        self.data_shape = (batch_size,) + data_shape
        self.label_shape = (batch_size,)
        self.provide_data = [('data', self.data_shape)]
        self.provide_label = [('softmax_label', self.label_shape)]
        self.batch = mx.io.DataBatch(data=[mx.nd.zeros(self.data_shape)],
                                     label=[mx.nd.zeros(self.label_shape)])
        self._batches = 0
        self.batches = batches

    def next(self):
        if self._batches < self.batches:
            self._batches += 1
            return self.batch
        else:
            self._batches = 0
            raise StopIteration


def dummy_iterator(batch_size, data_shape):
    return DummyIter(batch_size, data_shape), DummyIter(batch_size, data_shape)


class DummyImagenet(Imagenet):
    name = "dummy"
    num_examples = 1000

    def get_data(self, opt):
        return dummy_iterator(opt.batch_size, self.get_shape(opt)[1:])
