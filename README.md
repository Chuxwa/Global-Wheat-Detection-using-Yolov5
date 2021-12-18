# Global-Wheat-Detection
An efficient PyTorch library for Global Wheat Detection using [YOLOv5](https://github.com/ultralytics/yolov5). 
The project is based on this Kaggle competition [Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection) (May-Aug 2020).

**Here's a description of the prediction task**:

>In this competition, you’ll detect wheat heads from outdoor images of wheat plants, including wheat datasets from around the globe. Using worldwide data, you will focus on a generalized solution to estimate the number and size of wheat heads. To better gauge the performance for unseen genotypes, environments, and observational conditions, the training dataset covers multiple regions. You will use more than 3,000 images from Europe (France, UK, Switzerland) and North America (Canada). The test data includes about 1,000 images from Australia, Japan, and China.



## Installation

1. Create a virtual environment via `conda`.

   ```shell
   conda create -n wheat_detection python=3.7
   conda activate wheat_detection
   ```

2. Install `torch` and `torchvision`.

   ```shell
   conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
   ```

3. Install requirements.

   ```shell
   pip install -r requirements.txt
   ```

## Dataset
An overview is available here: https://www.kaggle.com/c/global-wheat-detection/data. 

Wheat heads were from various sources:  
<a href="https://imgur.com/HhOQtba"><img src="https://imgur.com/HhOQtba.jpg" title="head" alt="head" /></a>  
A few labeled images are as shown: (Blue bounding boxes)  
<a href="https://imgur.com/QhnuEEf"><img src="https://imgur.com/QhnuEEf.jpg" title="head" alt="head" width="378" height="378" /></a> <a href="https://imgur.com/5yUJCPV"><img src="https://imgur.com/5yUJCPV.jpg" title="head" alt="head" width="378" height="378" /></a>  

I used the following command to obtain the data:
```
$ kaggle competitions download -c global-wheat-detection
```

## Results






If you like our work and use the models for your research, please star our work.

## Acknowledgement
We thank for the inspiration from [YOLOv5](https://github.com/ultralytics/yolov5) and [SENet](https://github.com/hujie-frank/SENet)