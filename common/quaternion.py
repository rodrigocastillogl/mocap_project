# ---------------------------------------------- #
# Copyright (c) 2018-present, Facebook, Inc.
# https://github.com/facebookresearch/QuaterNet
# ---------------------------------------------- #

import torch
import numpy as np

# PyTorch based implementations of Quaternion methods

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Input
    ------
        * q : tensor with dimensions (N, 4) ; Quaternions
        * r : tensor with dimensions (N, 4) ; Quaternions

        N -> number of quaternions in the tensors

    Output
    ------
        * Tensor with dimensions (N ,4) ; Quaternions product
    """

    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # quaternions outer product
    terms = torch.bmm( r.view(-1, 4, 1 ), q.view(-1, 1, 4) )

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    return torch.stack( (w,x,y,z), dim = 1 ).view( original_shape )


def qrot(q, v):
    """
    Rotate vector(s) v about the rotations described by quaternion(s) q
    Input
    ------
        * q : tensor with dimensions (N, 4) ; Quaternions
        * v : tensor with dimensions (N, 3) ; Vectors

        n -> number os quaternions/vectors in the tensors 
    Output
    ------
        * Tensor with dimension (N, 3) ; Vectors rotated
    """

    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]


    original_shape = list(v.shape)

    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross( qvec, v, dim = 1 )
    uuv = torch.cross( qvec, uv, dim = 1)

    return ( v + 2 * ( q[:,:1] * uv + uuv ) ).view( original_shape )


def qeuler(q, order, epsilon = 0):
    """
    Convert quaternion(s) q to Euler Angles 
    Input
    ------
        * q : Tensor with dimensions (N, 4) ;  Quaternions
        * order   : order of rotation in Euler angles
        * epsilon : avoid indeterminate result
    Output
    ------
        * Tensor with dimensions (N, 3) ;  Euler angles
    """

