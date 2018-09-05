import numpy as np
import cv2
image_dir = '/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_pytorch/VOC2012/VOCtrainval/VOC2012/JPEGImages/'
indices = open('/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_pytorch/VOC2012/VOCtrainval/VOC2012/ImageSets/Segmentation/train.txt', 'r').read().splitlines()

print len(indices)
print indices[0]
print ('{}{}'.format(image_dir, indices[0])+'.jpg')
_img = cv2.imread('{}{}'.format(image_dir, indices[0])+'.jpg')
print _img.shape