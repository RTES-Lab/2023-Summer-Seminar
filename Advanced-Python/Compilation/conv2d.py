import numpy as np
from numba import jit, prange
import ctypes

# #####################
# Pure Python
# #####################

def conv2d_naive(x: np.ndarray, k: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Input data shape must be 2-dimensional.")
    if k.ndim != 2:
        raise ValueError("Kernel data shape must be 2-dimensional.")
    
    hx, wx = x.shape
    hk, wk = k.shape

    if hk > hx or wk > wx:
        raise ValueError("Kernel must not be larger than input data.")
    
    dst = np.zeros((hx-hk+1, wx-wk+1))
    
    for i in range(hx-hk+1):
        for j in range(wx-wk+1):
            for m in range(hk):
                for n in range(wk):
                    dst[i, j] += x[i+m, j+n] * k[m, n]

    return dst

def im2col(x: np.ndarray, filter_shape: np.ndarray):
    hx, wx = x.shape
    hk, wk = filter_shape[0], filter_shape[1]
    
    dst = np.zeros((hx-hk+1, wx-wk+1, hk, wk))

    for i in range(hx-hk+1):
        for j in range(wx-wk+1):
            for m in range(hk):
                for n in range(wk):
                    dst[i, j, m, n] = x[i+m, j+n]

    return dst.reshape((hx-hk+1)*(wx-wk+1), hk*wk)

def conv2d_unfold(x: np.ndarray, k: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Input data shape must be 2-dimensional.")
    if k.ndim != 2:
        raise ValueError("Kernel data shape must be 2-dimensional.")
    
    hx, wx = x.shape
    hk, wk = k.shape

    if hk > hx or wk > wx:
        raise ValueError("Kernel must not be larger than input data.")
    
    col = im2col(x, np.array([hk, wk]))
    dst = (np.dot(col, k.reshape(hk*wk, 1))).reshape(hx-hk+1, wx-wk+1)

    return dst

# #####################
# numba
# #####################

@jit(nopython=True)
def im2col_jit(x: np.ndarray, filter_shape: np.ndarray):
    hx, wx = x.shape
    hk, wk = filter_shape[0], filter_shape[1]
    
    dst = np.zeros((hx-hk+1, wx-wk+1, hk, wk))

    for i in range(hx-hk+1):
        for j in range(wx-wk+1):
            for m in range(hk):
                for n in range(wk):
                    dst[i, j, m, n] = x[i+m, j+n]

    return dst.reshape((hx-hk+1)*(wx-wk+1), hk*wk)

@jit(nopython=True)
def conv2d_naive_jit(x: np.ndarray, k: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Input data shape must be 2-dimensional.")
    if k.ndim != 2:
        raise ValueError("Kernel data shape must be 2-dimensional.")
    
    hx, wx = x.shape
    hk, wk = k.shape

    if hk > hx or wk > wx:
        raise ValueError("Kernel must not be larger than input data.")
    
    dst = np.zeros((hx-hk+1, wx-wk+1))
    
    for i in range(hx-hk+1):
        for j in range(wx-wk+1):
            for m in range(hk):
                for n in range(wk):
                    dst[i, j] += x[i+m, j+n] * k[m, n]

    return dst

@jit(nopython=True)
def conv2d_unfold_jit(x: np.ndarray, k: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Input data shape must be 2-dimensional.")
    if k.ndim != 2:
        raise ValueError("Kernel data shape must be 2-dimensional.")
    
    hx, wx = x.shape
    hk, wk = k.shape

    if hk > hx or wk > wx:
        raise ValueError("Kernel must not be larger than input data.")
    
    col = im2col_jit(x, np.array([hk, wk]))
    dst = (np.dot(col, k.ravel())).reshape(hx-hk+1, wx-wk+1)

    return dst

# #####################
# numba parallel
# #####################

@jit(nopython=True, parallel=True)
def im2col_pjit(x: np.ndarray, filter_shape: np.ndarray):
    hx, wx = x.shape
    hk, wk = filter_shape[0], filter_shape[1]
    
    dst = np.zeros((hx-hk+1, wx-wk+1, hk, wk))

    for i in prange(hx-hk+1):
        for j in prange(wx-wk+1):
            for m in prange(hk):
                for n in prange(wk):
                    dst[i, j, m, n] = x[i+m, j+n]

    return dst.reshape((hx-hk+1)*(wx-wk+1), hk*wk)

@jit(nopython=True, parallel=True)
def conv2d_naive_pjit(x: np.ndarray, k: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Input data shape must be 2-dimensional.")
    if k.ndim != 2:
        raise ValueError("Kernel data shape must be 2-dimensional.")
    
    hx, wx = x.shape
    hk, wk = k.shape

    if hk > hx or wk > wx:
        raise ValueError("Kernel must not be larger than input data.")
    
    dst = np.zeros((hx-hk+1, wx-wk+1))
    
    for i in prange(hx-hk+1):
        for j in prange(wx-wk+1):
            for m in range(hk):
                for n in range(wk):
                    dst[i, j] += x[i+m, j+n] * k[m, n]

    return dst

@jit(nopython=True, parallel=True)
def conv2d_unfold_pjit(x: np.ndarray, k: np.ndarray):
    if x.ndim != 2:
        raise ValueError("Input data shape must be 2-dimensional.")
    if k.ndim != 2:
        raise ValueError("Kernel data shape must be 2-dimensional.")
    
    hx, wx = x.shape
    hk, wk = k.shape

    if hk > hx or wk > wx:
        raise ValueError("Kernel must not be larger than input data.")
    
    col = im2col_jit(x, np.array([hk, wk]))
    dst = (np.dot(col, k.ravel())).reshape(hx-hk+1, wx-wk+1)

    return dst