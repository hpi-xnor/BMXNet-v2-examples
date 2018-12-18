# variable type for storing weights in binary inference layer
get_numpy_dtype = { 
	32 : 'uint32',
	64 : 'uint64'
}

get_mx_nd_dtype = { 
	32 : 'float32',
	64 : 'float64'
}

# layer name related
PREFIX_BINARIZED_FILE= 'binarized_'
POSTFIX_SYM_JSON = '-symbol.json'
PREFIX_Q_ACTIVATION = 'qactivation'
PREFIX_Q_CONV = 'qconv'
PREFIX_Q_DENSE = 'qdense'
PREFIX_WEIGHT ='_weight'

# symbol json related
PREFIX_SYM_JSON_NODES = 'nodes'
PREFIX_SYM_JSON_NODE_ROW_PTR = 'node_row_ptr'
PREFIX_SYM_JSON_ATTRS = 'attrs'
PREFIX_SYM_JSON_HEADS = 'heads'
PREFIX_SYM_JSON_ARG_NODES = 'arg_nodes'

# standard convolution and dense layer
PREFIX_DENSE = 'FullyConnected'
PREFIX_CONVOLUTION = 'Convolution'
# binary convolution and dense layer
binary_layer_replacements = { 
	PREFIX_CONVOLUTION : 'BinaryInferenceConvolution',
	PREFIX_DENSE 	   : 'BinaryInferenceFullyConnected'
}

# 2.for qconv and qdense, we only retain 'weight', 'bias' and 'fwd' operators   
retained_ops_patterns_in_conv_dense = ['weight', 'bias', 'fwd']

# use this to distinguish arg_nodes : op = 'null'
ARG_NODES_OP_PATTERN = 'null'