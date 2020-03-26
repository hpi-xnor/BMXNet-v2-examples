

class Dataset(object):
    name = ""
    num_classes = 0
    num_examples = 0
    default_save_frequency = 1
    shape = ()

    def get_shape(self, opt):
        return self.shape

    def get_data(self, opt):
        raise NotImplementedError
