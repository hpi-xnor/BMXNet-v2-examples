import mxnet as mx
import numpy as np

def set_bit(var, pos, val):
    '''
    description:
        this methods implements the following bit_set function:
        // variable, position, value
        #define BIT_SET(var, pos, val) var |= (val << pos)
    '''
    var |= (val << pos) 
    return var

def transpose_and_convert_to_binary_col(nd_in):
    '''
    description: 
        transpose and then binarize an array column wise
    '''
    nd_out = None
    return nd_out

def get_binary_row(nd_row, binary_row, nd_size, bits_per_binary_word):
    '''
    description:
        binarize the input NDArray. This is a reimplementation of the cpp version
        # for (int i = 0; i < size; i+=BITS_PER_BINARY_WORD) {
        #   BINARY_WORD rvalue=0;
        #   BINARY_WORD sign;
        #   for (int j = 0;j < BITS_PER_BINARY_WORD; ++j) {
        #     sign = (row[i+j]>=0);
        #     BIT_SET(rvalue, j, sign);
        #   }
        #   b_row[i/BITS_PER_BINARY_WORD] = rvalue;
        # }
    '''
    i = 0
    while i < nd_size:
        rvalue = 0        
        j = 0
        while j < bits_per_binary_word:
            if bits_per_binary_word == 32:
                sign = np.uint32(0)
            elif bits_per_binary_word == 64:
                sign = np.uint64(0)
            if nd_row[i+j] >= 0:
                sign = 1   
            set_bit(rvalue, j, sign)
            j += 1        
        binary_row[i/bits_per_binary_word] = rvalue
        i += bits_per_binary_word

    return binary_row

# def get_binary_col(nd_col, binary_col, dim_n, dim_k):