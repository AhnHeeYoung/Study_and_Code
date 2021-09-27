# Yolov3(2019)를 Pytorch로 구현한 코드입니다.

## Installation

### Clone and install requirements
```bash
$ git clone https://github.com/aladdinpersson/Machine-Learning-Collection
$ cd ML/Pytorch/object_detection/YOLOv3/
$ pip install requirements.txt
```

### Download pretrained weights on Pascal-VOC
Available on Kaggle: [link](https://www.kaggle.com/dataset/1cf520aba05e023f2f80099ef497a8f3668516c39e6f673531e3e47407c46694)

### Download Pascal VOC dataset
Download the preprocessed dataset from [link](https://www.kaggle.com/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video). Just unzip this in the main directory.

### Download MS COCO dataset
Download the preprocessed dataset from [link](https://www.kaggle.com/dataset/79abcc2659dc745fddfba1864438afb2fac3fabaa5f37daa8a51e36466db101e). Just unzip this in the main directory.

### Training
Edit the config.py file to match the setup you want to use. Then ```run train.py```
