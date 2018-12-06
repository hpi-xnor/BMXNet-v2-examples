import argparse
import mxnet as mx
import os
import logging
import sys
import json
from model_converter_config import *
from concatenation_operator import *


def get_output_files(model, output):
    '''
    model: name of the input model file
    output: output file path
    return: 
        - file name output symbol json file
        - file name of output params file
        - name of input json file
    description:
        prepare the file paths
    '''
    input_dir, file_name = os.path.split(model)
    out_params_file = PREFIX_BINARIZED_FILE + file_name

    k = out_params_file.rfind('-')
    out_json_file = out_params_file[:k] + POSTFIX_SYM_JSON
    
    k = file_name.rfind('-')
    input_json_file = os.path.join(input_dir, file_name[:k] + POSTFIX_SYM_JSON)

    if not output:
        output = input_dir
    out_json_file = os.path.join(output, out_json_file)
    out_params_file = os.path.join(output, out_params_file)

    return out_json_file, out_params_file, input_json_file

def convert_symbol_json(symbol):
    '''
    symbol: mxnet network symbol
    return: adapted json object
    description:
        Since it is not easy to create new graph for a symbol set.
        Go through the symbol internal items and change them is quite complicated, 
        we thus have to modify the json file.
        mxnet symbol json objects to be adapted:
        - arg_notes
        - heads
        - arg_nodes
        - node_row_ptr ? not yet found information about this item ?
    '''
    try:
        sym_items = json.loads(symbol.tojson())
    except Exception as e:
        raise Exception('load symbol.json failed')
    
    try:
        if PREFIX_SYM_JSON_NODES not in sym_items:
            raise Exception('"nodes" item not found in symbol.json file')
        if PREFIX_SYM_JSON_ARG_NODES not in sym_items:
            raise Exception('"arg_nodes" item not found in symbol.json file')
        if PREFIX_SYM_JSON_HEADS not in sym_items:
            raise Exception('"heads" item not found in symbol.json file')

        logging.info('"heads" of input json: %s' % sym_items[PREFIX_SYM_JSON_HEADS])
        logging.info('"arg_nodes" of input json: %s' % sym_items[PREFIX_SYM_JSON_ARG_NODES])

        # items we need to change
        nodes = sym_items[PREFIX_SYM_JSON_NODES]
        heads = sym_items[PREFIX_SYM_JSON_HEADS]
        arg_nodes = sym_items[PREFIX_SYM_JSON_ARG_NODES]

        nodes_new = []
        heads_new = [0, 0, 0]
        retained_op_num = 0
        # update arg_nodes : contains indices of all non-fwd nodes
        arg_nodes_new = []

        # update nodes
        for op in nodes:
            # 1. remove qactivation ops
            if PREFIX_Q_ACTIVATION in op['name']:
                continue
            
            # adapt qconv and qdense ops
            foundq = False
            if PREFIX_Q_CONV in op['name'] or PREFIX_Q_DENSE in op['name']:
                foundq = True
                retain = False
                # 2.for qconv and qdense, we only retain  'weight', 'bias' and 'fwd'                                
                for p in retained_ops_patterns_in_conv_dense:
                    if p in op['name']:
                        retain = True  
                # replace convolution and dense operators with binary inference layer
                if op['op'] == PREFIX_CONVOLUTION:
                    op['op'] = binary_layer_replacements[PREFIX_CONVOLUTION]
                    logging.info('converting op: {} from {} to {}'.format(op['name'], PREFIX_CONVOLUTION,
                                binary_layer_replacements[PREFIX_CONVOLUTION]))
                if op['op'] == PREFIX_DENSE:
                    op['op'] = binary_layer_replacements[PREFIX_DENSE]
                    logging.info('converting op: {} from {} to {}'.format(op['name'], PREFIX_DENSE,
                                binary_layer_replacements[PREFIX_DENSE]))                 
            if not foundq or retain:  
                # add node                
                nodes_new.append(op)
                # add arg_node
                if FWD_OP_PATTERN not in op['name']:
                    arg_nodes_new.append(retained_op_num)
                retained_op_num += 1        
                  
        # update heads 
        # heads : total num of nodes : [[index last element, 0, 0]]
        if retained_op_num > 0:
            heads_new[0] = retained_op_num - 1
        else:
            heads_new[0] = 0
        heads_new = [heads_new]
        sym_items[PREFIX_SYM_JSON_HEADS] = heads_new
        logging.info('update "heads" to: %s' % sym_items[PREFIX_SYM_JSON_HEADS])

        #update arg_nodes
        sym_items[PREFIX_SYM_JSON_ARG_NODES] = arg_nodes_new
        logging.info('update "arg_nodes" to: %s' % sym_items[PREFIX_SYM_JSON_ARG_NODES])

        # update nodes
        sym_items[PREFIX_SYM_JSON_NODES] = nodes_new
        #logging.info('update "nodes" to: %s' % sym_items[PREFIX_SYM_JSON_NODES])

    except Exception as e:
        raise e
    return sym_items
    

def convert_params(model_dict, bits):
    '''
    model_dict: model parameters in ndarray
    bits: depicts the bit number of applied binary word (uint32_t or uint64_t)

    description:
        concatinate the weights into binary_word.
        the standard 2D-Conv weight array dimension is 
        (output_dim, ipnut_dim, kernel_h, hernel_w)
        the standard dense layer weight array dimension is:
        (output_dim, input_dim)
    '''    
    for k, v in model_dict.items():
        tp, name = k.split(':', 1)
        # logging.info('{}:{}:{}'.format(tp, name, v.shape))
        # check dims
        if v.ndim < 2:
            continue

        if tp == 'arg' and PREFIX_Q_CONV in name and PREFIX_WEIGHT in name:
            
            logging.info('{}:{}:{}'.format(tp, name, v.shape[1]))             
            
            if v.shape[1] % bits != 0: # dim of input has to be divisible by bits (32 or 64)
                raise Exception('operator: "{}" has an invalid input dim: "{}", which is not divisible by 32 (or 64)'.format(name, v.shape[1]))

            size_binary_row = int(v.size / bits)
            # init binary row for concatenation
            binary_row = mx.nd.zeros((size_binary_row), dtype=get_dtype[bits])            
            binary_row = get_binary_row(v.reshape((-1)), binary_row, v.size, bits)

            # TODO: concatenate the qconv layer

        # if tp == 'arg' and PREFIX_Q_DENSE in name and PREFIX_WEIGHT in name:
        #     logging.info('{}:{}:{}'.format(tp, name, v.shape))              
            # TODO: concatenate the qdense layer
    

    


    return model_dict

def convert(model, output, bits):
    '''
    model: file name of input *.params model file
    output: output file directory path

    description:
        call the converting method for symbol json and *.params files
        Save the binarized model to the given output location.
    '''
    # get files
    out_json_file, out_params_file, input_json_file = get_output_files(model, output)

    # load and modify the symbol json file
    logging.info('============ convert *-symbol.json file: {} ============'.format(input_json_file))
    symbol = mx.sym.load(input_json_file)            
    sym_items = convert_symbol_json(symbol)
      
    # load params and convert mxnet *.params
    logging.info('============ convert *.params file: {} ============'.format(model))
    model_dict = mx.nd.load(model)
    model_dict_converted = convert_params(model_dict, bits)
    


    # try:
    #     logging.info('============ saving binarized params file: {} ============'.format(out_params_file))
    #     mx.nd.save(out_params_file, model_dict_converted)
    # except Exception as e:
    #     raise Exception('Save file "{}" failed.'.format(out_params_file))

    try:        
        logging.info('============ saving json symbol file: {} ============'.format(out_json_file))
        with open(out_json_file, 'w') as outfile:
            outfile.write(json.dumps(sym_items, indent=4, sort_keys=True))
            outfile.close()
    except Exception as e:
        raise Exception('Save file "{}" failed.'.format(out_json_file))


def check_bits(value):
    nbit = int(value)
    if nbit != 32 or nbit != 64:
         raise argparse.ArgumentTypeError("%s is an invalid value for '--bits', the valid value is 32 or 64" % value)
    return nbit
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description: model_converter will pack 32(x86 and ARMv7) or 64(x64) weight values into one and save the result with the prefix "binarized_"')
    parser.add_argument('--model', type=str, required=True, 
                        help = 'the input bmxnet *.params file')    
    parser.add_argument('--bits', type=check_bits, default=32, 
                        help = 'Valid value is 32 or 64. It defines the target binary_word bit-width used: 32(x86 and ARMv7) or 64(x64).')
    parser.add_argument('--output', type=str, default='', 
                        help = 'specify the location to store the binarized files. If not specified, the same location as the input model will be used.')
    args = parser.parse_args()
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # do it!
    convert(**vars(args))
