# Synchronized-BatchNormalization #

***Multi-Gpus Synchronized Batch Normalization implementation in PyTorch***

----------

## Introduction ##

This module is a synchronized version of Batch Normalization when using multi-gpus for deep learning, aka 'Syn-BN', as the mean and standard-deviation are reduced across all devices during training.

Traditionally, when using 'nn.DataParallel' to wrap module during training, the built-in PyTorch BatchNorm normalize the tensor on each device using the statistics only on that device, thus the statistics might be inaccurate. 

Instead, in this synchronized version, the statistics will be computed over all training samples distributed on each devices.

Besides, in single-gpu or cpu-only case, this module behaves exactly same as the built-in PyTorch implementation.

Note that this module may exist some design problems, if you have any questions or suggestions, please feel free to open an issue or submit a pull request, let's make it better!

----------


## Why Syn-BN ? ##

Usually, the working batch-size is typically large enough to obtain good statistics for some computer vision tasks, such as classification and detection, thus there is no need to synchronize BN layer during the training, while synchronization will slow down the training.

However, for the other computer vision tasks, such as semantic segmentation, which belongs to dense prediction problem, is very memory consuming, the working bath-size is usually very small(typically 2 or 4 in each GPU), thus it will hurt the performance without synchronization.

(*The importance of synchronized batch normalization in object detection has been proved with an extensive analysis in the paper [https://arxiv.org/abs/1711.07240](https://arxiv.org/abs/1711.07240 "MegDet: A Large Mini-Batch Object Detector")*)

----------

## How to use ?##

To use the Syn-BN, I customize a data parallel wrapper named 'DataParallelWithCallBack', which inherits nn.DataParallel, it will call a callback function when in data parallel replication. This introduces a slight difference with typical usage of the nn.DataParallel.

Use it with a provided, customized data parallel wrapper:

    from sync import DataParallelWithCallBack
    from sync_bn import SynchronizedBatchNorm2d
    
    sync_bn = SynchronizedBatchNorm2d(
        num_features=3, eps=1e-5, momentum=0.1, affine=True, sync_timeout=15.    
    )
    sync_bn = DataParallelWithCallBack(sync_bn, device_ids=[0, 1])
    sync_bn.to(device)

Or, if you have already defined a model wrapped in nn.DataParallel like:

    from torchvision import models
    
    m = models.resnet50(pretrained=True)
    m = nn.DataParallel(m, device_ids=[0,1])
    m.to(device)

then you can use the method 'convert_model' to convert your model to use Syn-BN easily:
    
    from func import convert_model

    m = convert_model(m)

this will change all BNs into Syn-BNs which is contained in your model.

----------

## Author ##

chrisway(cw), 2020.
