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
        * q : tensor with dimensions (N, 4) ; quaternion(s)
        * r : tensor with dimensions (N, 4) ; quaternion(s)

        N -> number of quaternions in the tensors

    Output
    ------
        * Tensor with dimensions (N ,4) ; quaternion(s) product
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
        * q : tensor with dimensions (N, 4) ; quaternion(s)
        * v : tensor with dimensions (N, 3) ; vector(s)

        n -> number os quaternions/vectors in the tensors 
    Output
    ------
        * Tensor with dimension (N, 3) ; vector(s) rotated
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
        * q : Tensor with dimensions (N, 4) ;  quaternion(s)
        * order   : order of rotation in Euler angles
        * epsilon : avoid indeterminate result
    Output
    ------
        * Tensor with dimensions (N, 3) ;  Euler angles
    """

    assert q.shape[-1] == 4
    
    original_shape = list(q.shape)
    original_shape[-1] = 3
    
    q = q.view(-1, 4)
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    
    if order == 'xyz':
        x = torch.atan2( 2 * (q0*q1 - q2*q3) , 1 - 2 * (q1*q1 + q2*q2) )
        y = torch.asin( torch.clamp( 2 * (q1*q3 + q0*q2), -1 + epsilon, 1 - epsilon) )
        z = torch.atan2( 2 * (q0*q3 - q1*q2), 1 - 2 * (q2*q2 + q3*q3) )
    elif order == 'yzx':
        x = torch.atan2( 2 * (q0*q1 - q2*q3) , 1 - 2 * (q1*q1 + q3*q3) )
        y = torch.atan2( 2 * (q0*q2 -q1*q3) , 1 - 2 * (q2*q2 + q3*q3) )
        z = torch.asin( torch.clamp( 2 * (q1*q2 + q0*q3), -1 + epsilon, 1- epsilon) )
    elif order == 'zxy':
        x = torch.asin( torch.clamp( 2 * (q0*q1 + q2*q3), -1 + epsilon, 1 - epsilon) )
        y = torch.atan2( 2 * (q0*q2 - q1*q3) , 1 - 2 * (q1*q1 + q2*q2) )
        z = torch.atan2( 2 * (q0*q3 - q1*q2), 1 - 2 * (q1*q1 + q3*q3) )
    elif order == 'xzy':
        x = torch.atan2( 2 * (q0*q1 + q2*q3), 1 - 2 * (q1*q1 + q3*q3) )
        y = torch.atan2( 2 * (q0*q2 + q1*q3), 1 - 2 * (q2*q2 + q3*q3) )
        z = torch.asin( torch.clamp(2 * (q0*q3 - q1*q2), -1 + epsilon, 1 - epsilon) )
    elif order == 'yxz':
        x = torch.asin( torch.clamp( 2 * (q0*q1 - q2*q3), -1 + epsilon, 1 - epsilon) )
        y = torch.atan2( 2 * (q1*q3 + q0*q2), 1 - 2 * (q1*q1 + q2*q2) )
        z = torch.atan2( 2 * (q1*q2 + q0*q3), 1 - 2 * (q1*q1 + q3*q3) )
    elif order == 'zyx':
        x = torch.atan2( 2 * (q0*q1 + q2*q3), 1 - 2 * (q1*q1 + q2*q2) )
        y = torch.asin( torch.clamp( 2 * (q0*q2 - q1*q3), -1 + epsilon , 1 - epsilon) )
        z = torch.atan2( 2 * (q0*q3 + q1*q2), 1 - 2 * (q2*q2 + q3*q3) )
    else:
        raise
        
    return torch.stack( (x,y,z), dim=1 ).view(original_shape)



# Numpy based implementations of Quaternion methods

def qmul_np(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Input
    ------
        * q : numpy array with dimensions (N, 4) ; quaternion(s)
        * r : numpy array with dimensions (N, 4) ; quaternion(s)

        N -> number of quaternions in the arrays

    Output
    ------
        * numpy array with dimensions (N ,4) ; quaternions product
    """
    
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()

    return qmul(q, r).numpy()


def qrot_np(q, v):
    """
    Rotate vector(s) v about the rotations described by quaternion(s) q
    Input
    ------
        * q : numpy array with dimensions (N, 4) ; quaternion(s)
        * v : numpy array with dimensions (N, 3) ; vector(s)

        n -> number os quaternions/vectors in the arrays 
    Output
    ------
        * numpy array with dimension (N, 3) ; vector(s) rotated
    """

    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()

    return qrot(q, v).numpy()


def qeuler_np(q, order, epsilon = 0, use_gpu = False):
    """
    Convert quaternion(s) q to Euler Angles 
    Input
    ------
        * q : numpy array with dimensions (N, 4) ;  quaternion(s)
        * order   : order of rotation in Euler angles
        * epsilon : avoid indeterminate result
        * use_gpu : flag
    Output
    ------
        * Tensor with dimensions (N, 3) ;  Euler angles
    """

    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()


# Numpy methods for exponential maps - quaternions - euler convertions


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representations (q or -q) with minimal distance between two
    consecutive frames
    Input
    ------
        * q : numpy array with dimensions (N, J, 4) ;  quaternions

        N -> number of quaternions (sequence length)
        J -> number of joints

    Output
    ------
        * numpy array with dimensions (N, J, 4); quaternions
    """

    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()

    # minimal euclidean distance is equivalent to minimal dot product
    dot_products = np.sum( q[1:]*q[:-1], axis = 2 )
    mask  = dot_products < 0
    mask =  ( np.cumsum(mask, axis = 0)%2 ).astype(bool)
    result[1:][mask] *= -1

    return result


def expmap_to_quaternion(e):
    """
    Convertion angle-axis rotations (exponential maps) to  quaternions.
    Fromula from:
    "Practical Parameterization of Rotations  Using the Exponential Map"
    Input
    ------
        * e : numpy array with dimensions (N, 3) ; angle-axis rotations

        N -> number of angle-axis rotations

    Output
    ------
        * numpy array with dimensions (N, 4); quaternions
    """

    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos( 0.5 * theta ).reshape(-1, 1)
    xyz = 0.5 * np.sinc( 0.5 * theta / np.pi ) * e

    return np.concatenate( (w, xyz), axis = 1 ).reshape(original_shape)


def euler_to_quaternion(e, order):
    """
    Convertion euler angles to quaternions.
    Fromula from:
    "Practical Parameterization of Rotations  Using the Exponential Map"
    Input
    ------
        * e : numpy array with dimensions (N, 3) ; euler angles
        * oder : order of rotation in Euler angles
        N -> number of rotations

    Output
    ------
        * numpy array with dimensions (N, 4); quaternions
    """

    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    
    e = e.reshape(-1, 3)
    
    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]
    
    rx = np.stack( ( np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x) ), axis = 1 )
    ry = np.stack( ( np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y) ), axis = 1 )
    rz = np.stack( ( np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2) ), axis = 1 )

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)
            
    # Reverse antipodal representation to have a non-negative w
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1
    
    return result.reshape(original_shape)