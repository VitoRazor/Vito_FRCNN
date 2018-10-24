# Vito_FRCNN
create an architecture for Faster_RCNN.
implementations 
## Table of Contents
  * [Installation](#installation)
  * [Detail](#Detail)
    + [Datasets](#Dataset)  
    + [GAN with Info](#GAN-info)

## Installation
    $ git clone https://github.com/VitoRazor/vito_FRCNN.git
    $ cd Vito_FRCNN-master/
    $ pip install keras

## Detail   
### Dataset
Implementation of this detector was trained on KITTI: http://www.cvlibs.net/datasets/kitti/eval_object.php
We set input as: 
    $ "imagepath","x","y","height","width","keyword"
such as:
    $ dataset/training/image_2/000001.png,387.63,181.54,423.81,203.12,Car

Result:
test in kitti testing  
 <p align="center">
    <img src="https://github.com/VitoRazor/Vito_FRCNN/blob/master/result/1.png" width="800"\>
</p>
 <p align="center">
    <img src="https://github.com/VitoRazor/Vito_FRCNN/blob/master/result/2.png" width="800"\>
</p>
 <p align="center">
    <img src="https://github.com/VitoRazor/Vito_FRCNN/blob/master/result/3.png" width="800"\>
</p>
 <p align="center">
    <img src="https://github.com/VitoRazor/Vito_FRCNN/blob/master/result/4.png" width="800"\>
</p>


