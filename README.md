# PSPNet-caffe
We recommend using [caffe-zlh](https://github.com/zhanglonghao1992/caffe-zlh)  
Thanks for [Soeaver](https://github.com/soeaver/caffe-model)'s job.  
To reduce memory usage, we merge all the models batchnorm layer parameters into scale layer. Using [mergr_bn_scale.py](PSPNet-caffe/tools/mergr_bn_scale.py) to do this.   
