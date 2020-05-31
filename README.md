# caffe_model2_tf_model

## attention: the ways of padding in tensorflow is not same with caffe
            
           for example: input_size: 224  stride: 2, kernel_size:3  padding:'same'
              
              in tensorflow:  if(input_size % stride == 0)   
                                  total_pad = max(kernel_size - stride, 0)
                                  top_pad = total_pad // 2 
                                  bottom_pad = total_pad - top_pad  
                                  output_size = (input_size + total_pad - kernel_size) // stride + 1 
                              else:
                                  total_pad = max(kernel_size - (input_size % stride), 0)
                                  top_pad = total_pad // 2
                                  bottom_pad = total_pad - top_pad 
                                  output_size = (input_size + total_pad - kernel_size) // stride + 1

              in caffe:  input_size: 224  stride:2  kernel_size:3   padding:1   
                        outpu_size = (input_size + 2 * padding - kernel_size)//stride + 1
               

## Reference:
    tensorflow padding: https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python

