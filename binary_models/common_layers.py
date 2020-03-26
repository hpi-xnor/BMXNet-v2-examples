from mxnet.gluon import nn


imagenet_variants = [
    "imagenet",
    "grouped_stem",
]


class ChannelShuffle(nn.HybridBlock):
    """
    ShuffleNet channel shuffle Block.
    """
    def __init__(self, groups=2, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self.groups = groups

    def hybrid_forward(self, F, x):
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        return data


def add_initial_layers(variant, seq_block, channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
    if variant == "thumbnail":
        seq_block.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, use_bias=False))
        return

    if norm_kwargs is None:
        norm_kwargs = {}

    if variant == "imagenet":
        seq_block.add(nn.Conv2D(channels, kernel_size=7, strides=2, padding=3, use_bias=False))
    elif variant == "grouped_stem":
        stem_width = channels // 2

        stem = nn.HybridSequential(prefix='stem_')
        with stem.name_scope():
            stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2, padding=1, use_bias=False))
            stem.add(norm_layer(in_channels=stem_width, **norm_kwargs))
            stem.add(nn.Activation('relu'))
            stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1, padding=1, groups=4, use_bias=False))
            stem.add(norm_layer(in_channels=stem_width, **norm_kwargs))
            stem.add(nn.Activation('relu'))
            stem.add(nn.Conv2D(channels=stem_width * 2, kernel_size=3, strides=1, padding=1, groups=8, use_bias=False))
        seq_block.add(stem)
    else:
        RuntimeError("Unknown initial layer variant: {}".format(variant))
    seq_block.add(norm_layer(**norm_kwargs))
    seq_block.add(nn.Activation('relu'))
    seq_block.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
