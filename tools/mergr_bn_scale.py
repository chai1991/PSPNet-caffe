import sys
sys.path.append('/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/caffe-priv/python')

import caffe
import numpy as np

caffe.set_mode_gpu()

net=caffe.Net('/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/pspnet/models/Resnet-101/deploy_resnet101-v2.prototxt',
              '/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/pspnet/models/Resnet-101/resnet101-v2.caffemodel',
              caffe.TEST)

new_net = caffe.Net('/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/pspnet/models/Resnet-101/deploy_resnet101-v2.prototxt',
                '/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/pspnet/models/Resnet-101/resnet101-v2.caffemodel',
                caffe.TEST)

for layer_name in net.params.keys():
    if layer_name[-2:] == 'bn':
        scale_layer_name = layer_name[:-2] + 'scale'
        mu = net.params[layer_name][0].data
        var = net.params[layer_name][1].data
        gamma = net.params[scale_layer_name][0].data
        beta = net.params[scale_layer_name][1].data
        new_gamma = gamma / (np.power(var, 0.5) + 1e-5)
        new_beta = beta - gamma * mu / (np.power(var, 0.5) + 1e-5)

        new_net.params[scale_layer_name][0].data[...] = new_gamma
        new_net.params[scale_layer_name][1].data[...] = new_beta

new_net.save('/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/pspnet/models/Resnet-101/resnet101_bn_scale_merged.caffemodel')
