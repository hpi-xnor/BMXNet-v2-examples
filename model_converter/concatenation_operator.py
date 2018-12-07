import mxnet as mx
import numpy as np
import ctypes

def bit_set(var, pos, val):
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
        binarize the input NDArray. 
        This is a re-implementation of the cpp version:
        for (int i = 0; i < size; i+=BITS_PER_BINARY_WORD) {
          BINARY_WORD rvalue=0;
          BINARY_WORD sign;
          for (int j = 0;j < BITS_PER_BINARY_WORD; ++j) {
            sign = (row[i+j]>=0);
            BIT_SET(rvalue, j, sign);
          }
          b_row[i/BITS_PER_BINARY_WORD] = rvalue;
        }
    '''
    i = 0
    while i < nd_size:
        rvalue = 0
        j = 0
        while j < bits_per_binary_word:
            sign = 0
            if nd_row[i+j] >= 0:
                sign = 1
            rvalue = bit_set(rvalue, j, sign)
            j += 1        
        
        print('rvalue after {0:64b}'.format(rvalue))
        # print("row before {}".format(binary_row[int(i/bits_per_binary_word)]))
        print('rvalue after {}'.format(rvalue))
        print(type(rvalue))
      
        binary_row[int(i/bits_per_binary_word)] = rvalue

        print('rvalue after {}'.format(int(binary_row[int(i/bits_per_binary_word)])))

        print(mx.nd.array(binary_row, dtype=np.float64)[int(i/bits_per_binary_word)])
        print('{0:.1f}'.format(mx.nd.array(binary_row, dtype=np.float64).asnumpy()[int(i/bits_per_binary_word)]))

        # print(binary_row.dtype)

        #print("row after {}".format(binary_row[int(i/bits_per_binary_word)]))
        # print('{0:32b}'.format(rvalue))
        # print('{0:32b}'.format(int(binary_row.asnumpy()[int(i/bits_per_binary_word)])))
        # print(int(binary_row.asnumpy()[int(i/bits_per_binary_word)]))


        i += bits_per_binary_word


        if i == 64:
            break        

    return binary_row

# def get_binary_col(nd_col, binary_col, dim_n, dim_k):
    '''
    description: 
        binarize an array column wise
    '''
    nd_out = None
    return nd_out