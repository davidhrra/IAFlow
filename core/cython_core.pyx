from cython.parallel cimport parallel, prange
from cython.view cimport array as cvarray
cimport cython
import multiprocessing
from libc.math cimport fabs

cdef int num_threads = multiprocessing.cpu_count() * 2

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef tuple maxPoolForward(double[:,:,:,:] x, int size, int stride):

    cdef Py_ssize_t new_shape = ((x.shape[2]-size)/stride)+1, examples = x.shape[0], filters = x.shape[3], original_shape = x.shape[1]
    prePooledArray = cvarray(shape=(examples, new_shape, new_shape, filters), itemsize=sizeof(double), format='d')
    cdef double[:,:,:,:] pooledArray = prePooledArray
    prePositions = cvarray(shape=(examples, filters, new_shape*new_shape, 2), itemsize=sizeof(int), format='i')
    cdef int[:,:,:,:] positions = prePositions
    cdef Py_ssize_t new_x = 0, new_y = 0, real_x = 0, real_y = 0, max_x = 0, max_y = 0, \
        positions_pos = 0, example = 0, filter = 0, i = 0, j = 0
    cdef double maximum = 0


    for example in range(examples):
        for filter in range(filters):
            for new_x in range(new_shape):
                for new_y in range(new_shape):
                    maximum = x[example, real_x, real_y, filter]
                    for i in range(size):
                        for j in range(size):
                            if x[example, real_x+i, real_y+j, filter] > maximum:
                                maximum = x[example, real_x+i, real_y+j, filter]
                                max_x = i
                                max_y = j
                    pooledArray[example, new_x, new_y, filter] = maximum
                    positions[example, filter, positions_pos, 0] = max_x+real_x
                    positions[example, filter, positions_pos, 1] = max_y+real_y
                    positions_pos += 1
                    real_y += stride
                real_y = 0
                real_x += stride
            real_x = 0
            positions_pos = 0

    return pooledArray, positions

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef double[:,:,:,:] maxPoolingBack(tuple objectiveArray, double[:,:,:,:] delta, int[:,:,:,:] positions):

    preFinalArray = cvarray(shape=objectiveArray, itemsize=sizeof(double), format='d')
    cdef double[:,:,:,:] finalArray = preFinalArray
    cdef Py_ssize_t examples = objectiveArray[0], filters = objectiveArray[3], Len = delta.shape[1]
    cdef Py_ssize_t example = 0, filter = 0, arrayPosition = 0, Xpos = 0, Ypos = 0, original_pos_x = 0, original_pos_y = 0

    for example in range(examples):
        for filter in range(filters):
            arrayPosition = 0
            for Xpos in range(Len):
                for Ypos in range(Len):
                    arrayPosition += 1
                    original_pos_x = positions[example, filter, arrayPosition, 0]
                    original_pos_y = positions[example, filter, arrayPosition, 1]
                    finalArray [example, original_pos_x, original_pos_y, filter] =\
                        delta[example, Xpos, Ypos, filter]

    return finalArray


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef double[:,:,:,:] convolution(double[:,:,:,:] x, double[:,:,:,:] kernel, int stride, tuple new_shape):
    cdef Py_ssize_t examples = new_shape[0], filters = new_shape[3], size = kernel.shape[0], channels = x.shape[3]
    preConvolvedX = cvarray(shape=new_shape, itemsize=sizeof(double), format='d')
    cdef double[:,:,:,:] convolved = preConvolvedX
    cdef double addition = 0
    cdef Py_ssize_t example = 0, k_filter = 0, channel = 0, new_x = 0, new_y = 0, m = 0, \
        n = 0, pointer_x = 0,  pointer_y = 0, c_counter = new_shape[1]-1


    convolved[:,:,:,:] = 0

    for example in prange(examples, nogil=True, num_threads=num_threads, schedule='dynamic'):
        for k_filter in range(filters):
            for channel in range(channels):
                for new_x in range(c_counter, -1, -1):
                    pointer_x = stride * new_x + size - stride
                    for new_y in range(c_counter, -1, -1):
                        pointer_y = stride * new_y + size - stride
                        addition = 0
                        for m in range(size):
                            for n in range(size):
                                addition += kernel[m, n, channel, k_filter] * \
                                            x[example, pointer_x-m, pointer_y-n, channel]
                        convolved[example, new_x, new_y, k_filter] += addition


    return convolved


''''@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef tuple backwardConvolution(double[:,:,:,:] x, double[:,:,:,:] delta, double[:,:,:,:] kernel, int stride):

    cdef tuple x_shape = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]), \
        kernel_shape = (kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3])

    cdef Py_ssize_t examples = x.shape[0], filters = kernel.shape[3], channels = kernel.shape[2], delta_size = delta.shape[1],\
        kernel_size = kernel.shape[0]
    cdef Py_ssize_t example = 0, channel = 0, pointer_x = 0, pointer_y = 0, y_delta = 0, \
        x_kernel = 0, y_kernel = 0, x_delta = 0, k_filter = 0
    pre_dX = cvarray(shape=x_shape, itemsize=sizeof(double), format='d')
    cdef double[:,:,:,:] dX = pre_dX
    pre_dW = cvarray(shape=kernel_shape, itemsize=sizeof(double), format='d')
    cdef double[:,:,:,:] dW = pre_dW
    dX[:,:,:,:] = 0
    dW[:,:,:,:] = 0


    for example in range(examples):
        for k_filter in range(filters):
            for channel in range(channels):
                pointer_x = 0
                for x_delta in range(delta_size):
                    pointer_y = 0
                    for y_delta in range(delta_size):
                        for x_kernel in range(kernel_size):
                            for y_kernel in range(kernel_size):
                                dX[example, pointer_x + x_kernel, pointer_y + y_kernel, channel] += kernel[x_kernel, y_kernel, channel, k_filter] * delta[example, x_delta, y_delta, k_filter]
                                dW[x_kernel, y_kernel, channel, k_filter] +=  x[example, pointer_x + x_kernel, pointer_y + y_kernel, channel] * delta[example, x_delta, y_delta, k_filter]
                        pointer_y += stride
                    pointer_x += stride

    return dX, dW'''

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef double[:,:,:,:] backwardConvolutionX(double[:,:,:,:] x, double[:,:,:,:] delta, double[:,:,:,:] kernel, int stride):
    cdef Py_ssize_t examples = x.shape[0], filters = kernel.shape[3], channels = kernel.shape[2], \
        delta_size = delta.shape[1], kernel_size = kernel.shape[0]

    cdef Py_ssize_t example = 0, k_filter = 0, channel= 0, x_delta = 0, y_delta = 0, x_kernel = 0, y_kernel = 0, \
        pointer_x = 0, pointer_y = 0

    cdef tuple x_shape = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]), \
        kernel_shape = (kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3])

    pre_dX = cvarray(shape=x_shape, itemsize=sizeof(double), format='d')
    cdef double[:,:,:,:] dX = pre_dX
    dX[:,:,:,:] = 0


    for example in prange(examples, nogil=True, num_threads=num_threads, schedule='dynamic'):
        for k_filter in range(filters):
            for channel in range(channels):
                for x_delta in range(delta_size):
                    for y_delta in range(delta_size):
                        for x_kernel in range(kernel_size):
                            for y_kernel in range(kernel_size):
                                pointer_x = stride * x_delta + x_kernel
                                pointer_y = stride * y_delta + y_kernel
                                dX[example, pointer_x, pointer_y, channel] += \
                                    kernel[x_kernel, y_kernel, channel, k_filter] * delta[example, x_delta, y_delta, k_filter]


    return dX


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef double[:,:,:,:] backwardConvolutionW(double[:,:,:,:] x, double[:,:,:,:] delta, int stride, tuple kernel_shape):

    cdef Py_ssize_t examples = x.shape[0], channels = x.shape[3], filters = delta.shape[3], \
        kernel_size = kernel_shape[1], l_size = delta.shape[2]
    preKernelArray = cvarray(shape=kernel_shape, itemsize=sizeof(double), format='d')
    cdef double [:,:,:,:] kernelArray = preKernelArray
    cdef Py_ssize_t example = 0, k_filter = 0, channel = 0, kernel_x = 0, kernel_y = 0, \
        l_x = 0, l_y = 0, pointer_x = 0, pointer_y = 0
    cdef double kernel_addition = 0
    kernelArray[:,:,:,:] = 0

    with nogil, parallel(num_threads = num_threads):
        for example in prange(examples, schedule='dynamic'):
            for k_filter in range(filters):
                for channel in range(channels):
                    for kernel_x in range(kernel_size):
                        for kernel_y in range(kernel_size):
                            kernel_addition = 0
                            pointer_x = kernel_x
                            for l_x in range(l_size):
                                pointer_y = kernel_y
                                for l_y in range(l_size):
                                    kernel_addition += delta[example, l_x, l_y, k_filter] * \
                                                   x[example, pointer_x, pointer_y, channel]
                                    pointer_y += stride
                                pointer_x += stride
                            kernelArray[kernel_x, kernel_y, channel, k_filter] += kernel_addition
    return kernelArray

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef double[:] gradientClipping(double[:] x, double threshold):
    cdef Py_ssize_t i = 0, x_size = x.shape[0]
    cdef double absolute = 0
    for i in prange(x_size, nogil =True, num_threads=num_threads, schedule='dynamic'):
        absolute = fabs(x[i])
        if absolute >= threshold:
            x[i] = (threshold/absolute) * x[i]
    return x