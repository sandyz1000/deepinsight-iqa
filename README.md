# Deepinsight-iqa 

## Technical Image quality analysis based on deep image assessment (DIQA) and Neural Image assessment (NIMA)

Content:
- DIQA (Deep Image Quality assesment) 
- NIMA (Neural Image Assesment)

What is DIQA?
-------------
DIQA is an original proposal that focuses on solving some of the most concerning challenges of applying deep learning to image quality assessment (IQA). The advantages against other methodologies are:
- The model is not limited to work exclusively with Natural Scene Statistics (NSS) images.
- Prevents overfitting by splitting the training into two phases (1) feature learning and (2) mapping learned features to subjective scores.

The cost of generating datasets for IQA is high since it requires expert supervision. Therefore, the fundamental IQA benchmarks are comprised of solely a few thousands of records. The latter complicates the creation of deep learning models because they require large amounts of training samples to generalize. 
The most frequently used datasets to train and evaluate IQA methods Live, TID2013, CSIQ. To add more image this package contains out of the box support for geometric augmentation (vertical/horizontal flips and random crops) which can create more dataset required for the DNN.


## Get Started
----------------
The network can be trained using custom network as well pretrained bottleneck layers. Currently this library support training with three bottleneck layer (Mobilenet, Inceptionv3 and Resnetv2), the diqa*.json in the confs directory can be configured for the respectives training methods.

**NOTE**: In case you want to add another bottleneck layers to this network make sure you add the proper scaling-factor
```
scaling_factor = 1/ 38. input_size = [416, 416], base_model_name = InceptionV3
scaling_factor = 1/ 38, input_size= [416, 416], base_model_name = InceptionResNetV2
scaling_factor = 1 / 32. , input_size = [416, 416], base_model_name = MobileNet
```

## Installation

Installation is easy!

```
pip install .
```


To begin Training the network
------------------------------
```
deepiqa_train -mdiqa \
-b/root/deepinsight-iqa/ -cconfs/diqa_mobilenet.json -fcombine.csv \
-i/root/dataset --train_model=all --load_model=subjective
```

For prediction:
--------------
```
deepiqa_predict -mdiqa \
-b/root/deepinsight-iqa/ -cconfs/diqa_conf.json -i/root/dataset/live/jp2k/img1.bmp 
```