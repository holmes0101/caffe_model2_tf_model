
import os
import tensorflow as tf 
import numpy as np 



def print_Layer(layer):
    print(layer.op.name, ' ', layer.get_shape().as_list())

def myConv2d(input_tensor, conv_size, stride_size ,output_channel, name, use_bias=False, regu=None, padding='SAME', act=tf.nn.relu, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        input_channel = input_tensor.get_shape()[-1].value
        weights = tf.get_variable(name='weights', shape=[conv_size, conv_size, input_channel, output_channel],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        
        conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=[1, stride_size, stride_size, 1], padding=padding,
                            use_cudnn_on_gpu=True, name=name)
        if regu != None and reuse != True:  ## reuse = False 当重用时，只计算一次
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regu)(weights))
        print_Layer(conv)

        if use_bias:
            biases = tf.get_variable(name='biases', shape=[output_channel], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.001))
            conv_bias = tf.nn.bias_add(value=conv, bias=biases, name='bias')
            print_Layer(conv_bias)
        else:
            conv_bias = conv 
        if act == None:
            return conv_bias 

        conv_relu = act(conv_bias, name='relu')
        print_Layer(conv_relu)
        return conv_relu

def conv_bn_relu(input_tensor, filters, kernel_size, strides, name, is_training=False, padding='same', activation=tf.nn.relu, use_bias=False, reuse=False):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, 
                               padding=padding,
                               activation=None,
                               use_bias=use_bias,
                               name=name + '_conv')(input_tensor)
    print_Layer(conv)
    bn = tf.layers.BatchNormalization(axis=3, trainable=is_training,
                                     momentum=0.9,
                                    name=name + '_bn')(conv)
    print_Layer(bn)                            
    if activation is not None:
        result = activation(bn, name=name + '_relu')
        print_Layer(result)
        return result 
    else:
        return bn 


def Net2(input_tensor, is_training=False):
    conv1 = myConv2d(input_tensor, conv_size=7, stride_size=2, 
                 output_channel=64, name='conv1', use_bias=True,
                 regu=None, padding='SAME', 
                 act=None, 
                 reuse=False)
    
    return conv1 

def Net(input_tensor, is_training=False):

    conv1 = myConv2d(input_tensor, conv_size=7, stride_size=2, 
                 output_channel=64, name='conv1', use_bias=True,
                 regu=None, padding='SAME', 
                 act=None, 
                 reuse=False)

    conv1_bn = tf.layers.BatchNormalization(axis=3, momentum=0.9,
                                            trainable=is_training,
                                            name='bn_conv1')(conv1)
    print_Layer(conv1_bn)
    conv1_bn_relu = tf.nn.relu(conv1_bn, name='conv1_relu')
    print_Layer(conv1_bn_relu)

    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                         padding='same', 
                                         name='pool1')(conv1_bn_relu)
    print_Layer(pool1)

    res2a_branch2a = myConv2d(pool1, conv_size=1, stride_size=1,
                              output_channel=64, name='res2a_branch2a',
                              use_bias=False, regu=None,
                              padding='VALID',
                              act=None, reuse=False)

    bn2a_branch2a = tf.layers.BatchNormalization(axis=3, momentum=0.9,
                                                trainable=is_training,
                                                name='bn2a_branch2a')(res2a_branch2a)
    print_Layer(bn2a_branch2a)

    res2a_branch2a_relu = tf.nn.relu(bn2a_branch2a, name='res2a_branch2a_relu')
    print_Layer(res2a_branch2a_relu)

    res2a_branch2b = myConv2d(res2a_branch2a_relu, conv_size=3, stride_size=1,
                             output_channel=64, name='res2a_branch2b',
                             use_bias=False, regu=None,
                             padding='SAME', 
                             act=None, 
                             reuse=False)

    return res2a_branch2b 

def read_weights_from_caffe_txt(weights_txt_path):
    with open(weights_txt_path, 'r') as f:
        lines = f.readlines()
    
    tf_weights_dict = {}
    for each_line in lines:
        each_line = each_line.rstrip()
        each_line_list = each_line.split(' ')

        ## layer name
        layer_name_str = each_line_list[0]
        layer_name = layer_name_str.split(':')[1]

        ## shape 
        shape_str = each_line_list[1]
        shape_str = shape_str.split(':')[-1]
        shape_list = [int(x) for x in shape_str.split('_')]
        shape_len = len(shape_list)

        ## weights 
        weights = [np.float32(x) for x in each_line_list[2:]]
        weights_np = np.array(weights, dtype=np.float32)
        
        if shape_len == 1:
            weights_tf = weights_np
        elif shape_len == 4:
            weights_np = np.reshape(weights_np, (shape_list[0], shape_list[1], shape_list[2], shape_list[3]))
            weights_tf = weights_np.transpose((2, 3, 1, 0))
        else:
            raise("weights shape is not support")
        
        tf_weights_dict[layer_name] = weights_tf
    return tf_weights_dict

def difference(tensor_np1, tensor_np2):
    diff = np.array(tensor_np1) - np.array(tensor_np2) 

    return np.sum(diff)


def get_inputs_from_txt(random_input_txt_path, input_size, input_channel=3):
    random_inputs = []
    with open(random_input_txt_path, 'r') as f:
        lines = f.readlines()
    
    for each_line in lines:
        each_line = each_line.rstrip()
        
        each_line_list = each_line.split(' ')
        each_input_tensor = [np.float32(x) for x in each_line_list]
        each_input_tensor_np = np.array(each_input_tensor, dtype=np.float32)
        each_input_tensor_np = np.reshape(each_input_tensor_np, (1, input_size, input_size, input_channel))
        random_inputs.append(each_input_tensor_np)
    
    return random_inputs

if __name__ == "__main__":

    input_size = 224 
    input_channel = 3 

    output_size = 112 
    output_channel = 64

    weights_txt_path = 'res50_part_weights.txt' ## this weights from caffemodel, store with channel first 
    random_input_txt_path ='./res50_part_random_input.txt'
    output_from_caffe_txt_path = './res50_part_output.txt'

    output_name = 'res2a_branch2b/res2a_branch2b:0'
    output_name = 'conv1/conv1:0'
    with tf.variable_scope('input', reuse=False):
        inputs = tf.placeholder(tf.float32, shape=[None, input_size, input_size, input_channel], name='inputs')
    
    random_inputs = get_inputs_from_txt(random_input_txt_path, input_size, input_channel)
    output_from_caffe_list = get_inputs_from_txt(output_from_caffe_txt_path, output_size, output_channel)

    output = Net2(inputs, is_training=False)

    tf_weights_dict = read_weights_from_caffe_txt(weights_txt_path)
    print(tf_weights_dict.keys())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        var_list = tf.global_variables()

        for var in var_list:
            print(var.name)
            print('before:')
            weights = sess.graph.get_tensor_by_name(var.name)
            print(weights.shape)
            print(weights)

            before_weights = sess.run(weights)
            ## assign weights 
            layer_name = var.name.split('/')[0]
            if 'bn' in var.name:
                layer_name_variance = layer_name + '_variance'
                layer_name_sc = layer_name.replace('bn', 'scale')
                layer_name_beta = layer_name_sc + '_beta'

                if 'moving_mean' in var.name:
                    sess.run(tf.assign(weights, tf_weights_dict[layer_name]))
                    up_weights = sess.run(weights)
                
                elif 'moving_variance' in var.name:
                    sess.run(tf.assign(weights, tf_weights_dict[layer_name_variance]))
                    up_weights = sess.run(weights)
                
                elif 'gamma' in var.name:
                    sess.run(tf.assign(weights, tf_weights_dict[layer_name_sc]))
                    up_weights = sess.run(weights)
                
                elif 'beta' in var.name:
                    sess.run(tf.assign(weights, tf_weights_dict[layer_name_beta]))
                    up_weights = sess.run(weights)
                
                else:
                    raise("bn not support this weights")
            elif ('conv' in var.name) or ('res' in var.name):
                layer_name = var.name.split('/')[0]
                if 'bias' in var.name:
                    layer_name_bias = layer_name + '_bias'
                    sess.run(tf.assign(weights, tf_weights_dict[layer_name_bias]))

                    up_weights = sess.run(weights)
                else:
                    sess.run(tf.assign(weights, tf_weights_dict[layer_name]))
                    up_weights = sess.run(weights)
            
            if difference(before_weights, up_weights) != 0:
                print("{} weights update Success".format(var.name))
            else:
                print("{} weights updata Failed".format(var.name))

    

        print(var_list)
        print(len(var_list))

        ## inference with new weights 
        test_input_num = len(random_inputs)
        output_list = []
        output_tensor = sess.graph.get_tensor_by_name(output_name)
        for each_idx in range(test_input_num):
            each_input_tensor = random_inputs[each_idx]
            each_output_tensor = sess.run(output_tensor, feed_dict={inputs: each_input_tensor})
            
            output_list.append(each_output_tensor)
        print(output_list)
        if difference(output_list, output_from_caffe_list) == 0:
            print("Varify caffe to Tensorflow weights Success")
        else:
            print("test input len:{}".format(len(output_list)))
            max_abs = np.max(np.abs(np.array(output_list) - np.array(output_from_caffe_list)))
            print("max_abs:{}".format(max_abs))
            print(difference(output_list, output_from_caffe_list))
            print("Varify caffe to Tensorflow weights Failed")


    