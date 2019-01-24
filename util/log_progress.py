import os
import math

intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
)


def display_time(seconds, granularity=2):
    result = []

    seconds = int(round(seconds))

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])


def log_progress(num_examples, opt, epoch, i, time_this_epoch, prev_epochs_avg):
    if i == 0:
        return
    if not os.path.isdir(os.path.dirname(opt.progress)):
        return
    its_per_epoch = math.ceil(num_examples / opt.batch_size)
    epoch_total_est = int(round((time_this_epoch * its_per_epoch) / i))
    eta_epoch = display_time(epoch_total_est - time_this_epoch)
    full_epochs_left = opt.epochs - epoch - 1
    if full_epochs_left < 0:
        full_epochs_left = 0
    if prev_epochs_avg > 0:
        epoch_total_est = prev_epochs_avg
    eta_train = display_time(epoch_total_est - time_this_epoch + full_epochs_left * epoch_total_est)

    with open(opt.progress, "w") as handle:
        handle.write("Epoch ETA: {}, Train ETA: {}".format(eta_epoch, eta_train))
