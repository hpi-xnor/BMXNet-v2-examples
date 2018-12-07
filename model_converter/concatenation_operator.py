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
        
        # print('{0:64b}'.format(rvalue))
        
        binary_row[int(i/bits_per_binary_word)] = rvalue
        
        # print('{0:64b}'.format(binary_row[int(i/bits_per_binary_word)]))
        # testing stuff
        # d = mx.nd.array(binary_row, dtype="float64")
        # print('{0:64b}'.format(int(d.asnumpy()[int(i/bits_per_binary_word)])))
        i += bits_per_binary_word
    return binary_row

def get_binary_col(nd_col, binary_col, dim_n, dim_k, bits_per_binary_word):
    '''
    description: 
        binarize an array column wise.
        A re-implementation of the cpp version:

        for(int y=0; y<(n/BITS_PER_BINARY_WORD); y++){
          for(int x=0; x < k; ++x){          
            BINARY_WORD rvalue=0;
            BINARY_WORD sign;    
            for(int b=0; b<BITS_PER_BINARY_WORD; ++b){
              sign = (col[(y*BITS_PER_BINARY_WORD+b)*k + x]>=0);          
              BIT_SET(rvalue, b, sign);
            }
            b_col[y*k + x] = rvalue;
          }
        }   

    '''
    y = 0
    while y < int(dim_n/bits_per_binary_word):
        x = 0
        while x < dim_k:
            rvalue = 0            
            b = 0
            while b < bits_per_binary_word:
                sign = 0
                if nd_col[(y*bits_per_binary_word+b)*dim_k + x] >= 0:
                    sign = 1
                bit_set(rvalue, b, sign)
                b+=1
            binary_col[y*dim_k + x] = rvalue
            x+=1
        y+=1

    return binary_col