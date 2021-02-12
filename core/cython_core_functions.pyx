from cython.parallel cimport parallel, prange
from cython.view cimport array as cvarray
cimport cython
import multiprocessing

cdef double e = 2.718281828459045
cdef int num_threads = multiprocessing.cpu_count()

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef double[:] leakyReluDerivative(double[:] x, double alpha):
    cdef int xLen = x.shape[0], i = 0
    with nogil, parallel(num_threads = num_threads):
        for i in prange(xLen):
            if x[i] >= 0:
                x[i] = 1.0
            else:
                x[i] = alpha
    return x

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef double[:] sigmoid(double[:] x):
    cdef int xLen = x.shape[0], i = 0
    with nogil, parallel(num_threads = num_threads):
        for i in prange(xLen):
            x[i] = 1 / (1 + e ** (-x[i]))
    return x

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef double[:] leakyReLU(double[:] x, double alpha):
    cdef int xLen = x.shape[0], i = 0
    with nogil, parallel(num_threads = num_threads):
        for i in prange(xLen):
            if x[i] >= 0:
                x[i] = x[i]
            else:
                x[i] = alpha * x[i]
    return x



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef double[:,:] softmax(double[:,:] x):
    cdef int j = 0, examples = x.shape[0], example = 0, units = x.shape[1]
    prex_return = cvarray(shape=(examples, units), itemsize=sizeof(double), format='d')
    pre_sum = cvarray(shape=(examples,1), itemsize=sizeof(double), format='d')
    cdef double [:,:] x_return = prex_return
    cdef double[:,:] i_sum = pre_sum
    cdef double result = 0
    i_sum[:,:] = 0
    with nogil, parallel(num_threads=num_threads):
        for example in prange(examples):
            for j in range(units):
                result = e ** x[example,j]
                x_return[example, j] = result
                i_sum[example,0] += result
            for j in range(units):
                x_return[example, j] /= i_sum[example,0]
    return x_return
