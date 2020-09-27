import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import math


def simple_mean_filter(gradients, net):
    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    grad_collect = []
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            # append to list for uploading to cloud
            grad_collect.append(mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size
    return grad_collect


def cgc_filter(gradients, net, f):
    """not finished"""
    euclidean_distance = []
    for i, x in enumerate(gradients):
        print(i)
        # euclidean_distance.append((i, nd.norm(nd.array(x))))
        euclidean_distance.append((i, np.linalg.norm(x)))
    gradients = [gradients[x[0]] for x in sorted(euclidean_distance, key=lambda x: x[1], reverse=True)[f:]]

    # print(len(gradients))

    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    mean_nd = nd.sum(nd.concat(*param_list, dim=1), axis=-1)
    grad_collect = []
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            # append to list for uploading to cloud
            grad_collect.append(mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size
    return grad_collect