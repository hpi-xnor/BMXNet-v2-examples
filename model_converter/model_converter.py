import argparse
import mxnet as mx
import os
import logging
import sys
import json

PREFIX_BINARIZED_FILE= 'binarized_'
POSTFIX_SYM_JSON = '-symbol.json'
PREFIX_Q_ACTIVATION = 'qactivation'
PREFIX_Q_CONV = 'qconv'
PREFIX_Q_DENSE = 'qdense'
PREFIX_REPLACE_CONV_LAYER = ''
PREFIX_REPLACE_DENSE_LAYER = ''

list_q_layers = ['qactivation', 'qconv', 'qdense']


def get_output_files(model, output):
    '''
    prepares the output file saving path
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

def load_params(model_dict):
    for k, v in model_dict.items():
        tp, name = k.split(':', 1)
        print('{}:{}:{}'.format(tp, name, v.shape))
        
        '''
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
        '''    

def convert(model, output):
    # get files
    out_json_file, out_params_file, input_json_file = get_output_files(model, output)

    # modify the symbol json file
    symbol = mx.sym.load(input_json_file)    
    sym_edited = []
    
    for op in symbol.get_internals():
        # 1.we directly remove qactivation layer
        if PREFIX_Q_ACTIVATION in op.name:
            continue

        # 2.for qconv and qdense, we only preserve 'weight', 'bias' and 'fwd'                
        s_patterns = ['weight', 'bias', 'fwd']
        foundq = False
        if PREFIX_Q_CONV in op.name or PREFIX_Q_DENSE in op.name:
            foundq = True
            retain = False
            for p in s_patterns:
                if p in op.name:
                    retain = True                
        if not foundq or retain:
            sym_edited.append(op.name)
    
    # since mx.sym.Group not work, it can not create new graph for a syms set.
    # so we have to edit the json file
    for op in group.get_internals():        
        print(op)

    # arg_nodes : non fwd nodes
    # heads : total num of nodes : [[total_num, 0, 0]]




    #sym_json = json.loads(group.tojson())
    #logging.info(json.dumps(sym_json, indent=4, sort_keys=True))    

#    for op in sym_edited:#.get_internals():        
#        print(op)


    #logging.info(symbol.get_internals())
    #logging.info(symbol.list_outputs())
    
    # load params
    #model_dict = mx.nd.load(model)
    #load_params(model_dict)

    # concatenate the qconv layer

    # concatenate the qdense layer






    #logging.info('Saving output params file: {}'.format(out_params_file))
    #logging.info('Saving json symbol file: {}'.format(out_json_file))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description: model_converter will pack 32(x86 and ARMv7) or 64(x64) weight values into one and save the result with the prefix "binarized_"')
    parser.add_argument('--model', type=str, required=True, help = 'the input bmxnet *.params file')    
    parser.add_argument('--output', type=str, default='', help = 'specify the location to store the binarized files. If not specified, the same location as the input model will be used.')
    args = parser.parse_args()
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    convert(**vars(args))
