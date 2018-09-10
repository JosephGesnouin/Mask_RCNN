import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/footballplayers/"))  # To find local version
import footplayers

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_foot.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR,'../logs/footplayers20180711T1442/mask_rcnn_footplayers_0001.h5')
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images/football/")

class InferenceConfig(footplayers.footplayersConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'fp']

# Load a random image from the images folder
#file_names = next(os.walk(IMAGE_DIR))[2]
#filename = os.path.join(IMAGE_DIR, 'test.jpg') #os.path.join(IMAGE_DIR, random.choice(file_names))
filename = os.path.join('/Users/jzk/Documents/Games/toulouse_images/scene131521.jpg')
filename = os.path.join('/Users/jzk/Documents/Stage_Mines/Mask_RCNN/samples/footballplayers/E.jpg')
image = skimage.io.imread(filename)
results = model.detect([image], verbose=1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
"""for i in file_names:
    if i.endswith('.jpg'):
        filename = os.path.join(IMAGE_DIR,i)
        print("working on element"+i)
        image = skimage.io.imread(filename)
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
        """
