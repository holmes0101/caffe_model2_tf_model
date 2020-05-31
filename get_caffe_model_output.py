

import os 
import sys 
sys.path.insert(0, '/home/holmes/caffe/python')

import caffe 
import numpy as np 

def random_generate_input_save_txt(input_size, save_txt_path, channel=3, num=10):

    writer = open(save_txt_path, 'w')
    inputs = []
    for idx in range(num):
        each_input = np.random.random([1, input_size, input_size, channel])
        inputs.append(each_input)
        batch = 1
        total_num = batch * input_size * input_size * channel 
        counter = 0
        for b_idx in range(batch):
            for h_idx in range(input_size):
                for w_idx in range(input_size):
                    for c_idx in range(channel):
                        counter += 1
                        if counter == total_num:
                            writer.writelines('{:.16}'.format(each_input[b_idx, h_idx, w_idx, c_idx]) + '\n')
                        else:
                            writer.writelines('{:.16}'.format(each_input[b_idx, h_idx, w_idx, c_idx]) + ' ')
    writer.close()

    return inputs 

def save_network_result(output_result_list, save_txt_path):
    writer = open(save_txt_path, 'w')
    for idx in range(len(output_result_list)):
        each_result = output_result_list[idx]
        batch, height, width, channel = each_result.shape
        total_num = batch * height * width * channel
        counter = 0
        for b_idx in range(batch):
            for h_idx in range(height):
                for w_idx in range(width):
                    for c_idx in range(channel):
                        counter += 1
                        if counter == total_num:
                            writer.writelines(str(each_result[b_idx, h_idx, w_idx, c_idx]) + '\n')
                        else:
                            writer.writelines(str(each_result[b_idx, h_idx, w_idx, c_idx]) + ' ')
    writer.close()


def get_output_list(net, inputs_list, output_name='reres2a_branch2bs'):
    output_list = []
    for idx in range(len(inputs_list)):
        each_random_input = inputs_list[idx]
        each_random_input_channel_first = each_random_input.transpose((0, 3, 1, 2))
        net.blobs['data'].data[...] = each_random_input_channel_first
        output = net.forward()

        each_result = output[output_name]
        each_result = np.array(each_result)
        each_result_channel_last = each_result.transpose((0, 2, 3, 1))

        output_list.append(each_result_channel_last)
        # print(each_result_channel_last)
        # print(each_result_channel_last.shape)
        # raise('fafd')
    
    return output_list


def save_weights2_txt(layer_name, weights, save_weights_txt_path):
    
    if weights is None:
        return
    
    if os.path.exists(save_weights_txt_path):
        writer = open(save_weights_txt_path, 'a+')
    else:
        writer = open(save_weights_txt_path, 'w')
    weights_shape = weights.shape 
    
    ## writer layer_name 
    writer.writelines('Layer_name:{} shape:'.format(layer_name))
    ## writer shape 
    shape_str = ''
    for shape_idx in range(len(weights_shape)):
        if shape_idx == (len(weights_shape) - 1):
            shape_str = shape_str + str(weights_shape[shape_idx]) + ' '
        else:
            shape_str = shape_str + str(weights_shape[shape_idx]) + '_'
    writer.writelines(shape_str)

    ## writer weights 
    if len(weights_shape) == 4:
        output_channel, kernel_h, kernel_w, input_channel = weights_shape
        total_num = output_channel * kernel_h * kernel_w * input_channel
        counter = 0 
        for o_idx in range(output_channel):
            for h_idx in range(kernel_h):
                for w_idx in range(kernel_w):
                    for i_idx in range(input_channel):
                        counter += 1
                        if counter == total_num:
                            writer.writelines('{:.16}'.format(weights[o_idx, h_idx, w_idx, i_idx]) + '\n')
                        else:
                            writer.writelines('{:.16}'.format(weights[o_idx, h_idx, w_idx, i_idx]) + ' ')
                        
    elif len(weights_shape) == 1:
        num = weights_shape[0]
        for idx in range(num):
            if idx == (num - 1):
                writer.writelines('{:.16}'.format(weights[idx]) + '\n')
            else:
                writer.writelines('{:.16}'.format(weights[idx]) + ' ')
    else:
        raise("weights_shape:{} not supoort".format(weights_shape))
    
    writer.close() 

def save_weights(net, save_weights_txt_path):
    
    for layer_name, layer_pointer in net.params.items():
        
        if 'bn' in layer_name:
            weights = net.params[layer_name][0].data 
            bias = net.params[layer_name][1].data 
            layer_name_bias = layer_name + '_variance'
        elif 'scale' in layer_name:
            weights = net.params[layer_name][0].data
            bias = net.params[layer_name][1].data 
            layer_name_bias = layer_name + '_beta'
        elif ('conv' in layer_name) or ('res' in layer_name):
            weights = net.params[layer_name][0].data   
            try:
                bias = net.params[layer_name][1].data    
            except:
                bias = None   
            
            layer_name_bias = layer_name + '_bias'
        else:
            print("layer_name:{} is not support now".format(layer_name))
            raise("layer name support")
        
        save_weights2_txt(layer_name, weights, save_weights_txt_path)
        save_weights2_txt(layer_name_bias, bias, save_weights_txt_path)

if __name__ == "__main__":
    ori_caffemodel_path = './ResNet-50-model.caffemodel'
    new_depoly_path = './deploy_test.prototxt'
    new_depoly_path = './deploy_test_just_conv.prototxt'

    input_size = 224 

    save_random_input_txt_path ='./res50_part_random_input.txt'
    save_output_txt_path = './res50_part_output.txt'

    save_weights_txt_path = './res50_part_weights.txt'

    if os.path.exists(save_weights_txt_path):
        rm_cmd = 'rm -rf {}'.format(save_weights_txt_path)
        os.system(rm_cmd)

    inputs_list = random_generate_input_save_txt(input_size, save_random_input_txt_path, num=10)
    caffe.set_mode_gpu()

    net = caffe.Net(new_depoly_path, ori_caffemodel_path, caffe.TEST)

    ## save caffe output as txt: channel last(like tensorflow)  
    output_list = get_output_list(net, inputs_list, output_name='conv1')
    save_network_result(output_list, save_output_txt_path)

    ## save caffe weights 
    save_weights(net, save_weights_txt_path)