import sys
sys.path.append('/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/caffe-priv/python')

import caffe
import numpy as np

caffe.set_mode_gpu()

net=caffe.Net('/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/pspnet/models/Resnet-101/deploy_resnet101-v2.prototxt',
              '/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/pspnet/models/Resnet-101/resnet101-v2.caffemodel',
              caffe.TEST)

net_merged=caffe.Net('/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/pspnet/models/Resnet-101/deploy_resnet101-v2.prototxt',
              '/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/pspnet/models/Resnet-101/resnet101_bn_scale_merged.caffemodel',
              caffe.TEST)


for layer_name in net.params.keys():
    if layer_name == 'res1_conv1_scale':
        print net.params[layer_name][0].data

for layer_name in net_merged.params.keys():
    if layer_name == 'res1_conv1_scale':
        print net_merged.params[layer_name][0].data

