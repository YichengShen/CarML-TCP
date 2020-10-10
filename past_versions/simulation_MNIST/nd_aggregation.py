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
        norms = [nd.norm(p) for p in x]
        norm_product = 1
        for each in norms:
            norm_product *= float(each.asnumpy()[0])
        euclidean_distance.append((i, norm_product))
    # euclidean_distance = sorted(euclidean_distance, key=lambda x: x[1], reverse=True)
    # output = []
    # for i in range(f, len(gradients)):
    #     output.append(gradients[euclidean_distance[i][0]])
    output = [gradients[x[0]] for x in sorted(euclidean_distance, key=lambda x: x[1], reverse=True)[f:]]


    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in output]
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