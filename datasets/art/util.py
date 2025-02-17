import logging
from colorlog import ColoredFormatter
import os
import numpy as np
import torch


ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
#    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('rn')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_train_and_test_loaders(
    path,
    img_size,
    batch_size,
    n_shapes,
    m_holdout,
    train_set_size,
    test_set_size,
    train_gen_method,
    test_gen_method,
    train_proportion,
    dataset_cls,
    create_task_fn,
    transformations=None,
):
    all_shapes = np.arange(n_shapes)
    np.random.shuffle(all_shapes)

    if m_holdout > 0:
        train_shapes = all_shapes[m_holdout:]
        test_shapes = all_shapes[:m_holdout]
    else:
        train_shapes = all_shapes
        test_shapes = all_shapes

    train_set, test_set = create_task_fn(
        n_shapes, m_holdout, train_set_size, test_set_size, train_gen_method, test_gen_method, train_proportion,
        train_shapes, test_shapes)

    train_data = dataset_cls(path, train_set, img_size, n_shapes, transformations=transformations)
    test_data = dataset_cls(path, test_set, img_size, n_shapes, transformations=transformations)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, drop_last=False)
    return train_dataloader, test_dataloader