import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import math


def simple_mean(gradients, net, lr):
    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    grad_collect = []
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            # directly update model
            # param.set_data(param.data() - lr * mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            # append to list for uploading to cloud
            grad_collect.append(param.data() - lr * mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size
    return grad_collect