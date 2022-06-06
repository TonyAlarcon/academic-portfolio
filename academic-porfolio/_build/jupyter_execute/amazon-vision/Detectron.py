#!/usr/bin/env python
# coding: utf-8

# # Detectron2

# In[1]:


#!pip install pyyaml==5.1

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# Install detectron2 that matches the above pytorch version
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
get_ipython().system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html')
# If there is not yet a detectron2 release that matches the given torch + CUDA version, you need to install a different pytorch.

#exit(0)  # After installation, you may need to "restart runtime" in Colab. This line can also restart runtime
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
get_ipython().system('pip install opencv-python')
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# In[2]:


#uncoment when working on Google Colab
from google.colab import drive
drive.mount("/content/gdrive")
print(os.getcwd())


# I used https://plainsight.ai to perform polygon annotations and exported the data utilizing their COCO format feature. However, the images segmentations are exported in the following format "segmentation" : [[x_1, y_1], [x_2, y_2], ...] which is not the forma expected from detectron2. Thefore, we first need to flatten the segmentation array of arrays prior to registering with detectron. The cell below takes care of that

# In[3]:


import json
import numpy as np


def convert_to_COCO(json_path):
  #variable where we will save our new json object
    new_json = None
    
    with open(json_path, 'r') as file:
        json_object = json.load(file)
        annotations = json_object['annotations']
    #function to flatten the array of arrays
    for i in range(0, len(annotations)):
        annotations[i]["segmentation"]  =  [np.array(annotations[i]["segmentation"]).flatten().tolist()]

    #save to json object 
    json_object['annotations'] = annotations

    new_json = json_object
    return new_json


#function to save a json file to selected path, note we should append the name of the file
def save_JSON(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


#Google colab path
#dataset_path = "/content/gdrive/MyDrive/Spring 2022 Courses/Neural Networks/Amazon-Robotics-snapshot-001-project-coco-1646593032"
dataset_path = '/content/gdrive/MyDrive/Amazon Project/clutterized/'
#scratch365 path
#dataset_path = "./Amazon-Robotics-snapshot-001-project-coco-1646593032"
old_annotations_train = dataset_path + "/train.json"
image_path = dataset_path 


new_json_train = convert_to_COCO(old_annotations_train)
save_JSON(new_json_train, dataset_path+"new_train.json")
annotations_train = dataset_path + "new_train.json"


# If we want to use a custom dataset while also reusing detectron2’s data loaders, you will need to:
# 
# 1.   Register your dataset (i.e., tell detectron2 how to obtain your dataset).
# 2.   Optionally, register metadata for your dataset.
# 
# It contains a mapping from strings (which are names that identify a dataset, e.g. “coco_2014_train”) to a function which parses the dataset and returns the samples in the format of list[dict]. 
# 
# For more details: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html

# In[4]:


from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode

#telling detectron what to call my training dataset, path to json file, and path to images
register_coco_instances("my_dataset_train", {}, annotations_train, image_path)

#visualize training data
train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset = DatasetCatalog.get("my_dataset_train") #Call the registered function and return its results (return list[dict] – dataset annotations.)


# Here we visualize random sample from our dataset, including the bbox, polygon segmentation and it's label

# In[5]:


import random
from detectron2.utils.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

rows, cols = 4, 3
j = 0
fig = plt.figure(figsize=(45,45))

for d in random.sample(train_dataset, 11): #returns 11 random samples from the training dataset
    img = cv2.imread(d["file_name"]) #respective file name
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.2)
    out = visualizer.draw_dataset_dict(d)
    fig.add_subplot(rows, cols, j+1)
    j = j+ 1
    plt.imshow(out.get_image()[:, :, ::-1])


plt.show()


# # Fine Tunning and Training

# In[6]:


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 600  # 600 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 200  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = '/content/gdrive/MyDrive/Amazon Project/clutterized/output/'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = CustomTrainer(cfg)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


# # Training Curves

# In[7]:


# Look at training curves in tensorboard:
#%load_ext tensorboard
#%tensorboard --logdir output

