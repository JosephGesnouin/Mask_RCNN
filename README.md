# Mask R-CNN for players segmentation in a [football game](https://www.youtube.com/watch?v=RQ97o6tM8gc&index=3&list=PLasxefpCczor6fWojQbdwGA0lMMCPzodp)
![Instance Segmentation Sample](assets/4k_video.gif)

## Installation
either follow [that](https://www.youtube.com/watch?v=2TikTv6PWDw) or the following steps
1. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
2. Clone this repository
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)
    To train or test on MS COCO, you'll also need:
         * [MS COCO Dataset](http://cocodataset.org/#home)
         * Download the 5K [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)
           and the 35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)
           subsets.

# Getting Started
* ([demo.ipynb](samples/demo.ipynb), [demoBallon.ipynb](samples/demoBallon.ipynb), [demoFoot.ipynb](samples/demoFoot.ipynb))  Is the easiest way to start. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images.
It includes code to run object detection and instance segmentation on arbitrary images.

* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask RCNN implementation. 

# Training on MS COCO
We're providing pre-trained weights for MS COCO to make it easier to start. You can
use those weights as a starting point to train your own variation on the network.
Training and evaluation code is in `samples/coco/coco.py`. You can import this
module in Jupyter notebook (see the provided notebooks for examples) or you
can run it directly from the command line as such:

```
# Train a new model starting from pre-trained COCO weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=last
```

You can also run the COCO evaluation code with:
```
# Run COCO evaluation on the last trained model
python3 samples/coco/coco.py evaluate --dataset=/path/to/coco/ --model=last
```

The training schedule, learning rate, and other parameters should be set in `samples/coco/coco.py`.


# Training on Your Own Dataset

Start by reading this [blog post about the balloon color splash sample](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46). It covers the process starting from annotating images to training to using the results in a sample application.


In summary, to train the model on your own dataset you'll need to extend two classes:

```Config```
This class contains the default configuration. Subclass it and modify the attributes you need to change.

```Dataset```
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

See examples in `samples/balloonfoot`, `samples/coco`, and `samples/footplayers.py`.



# how to train from scratch on 2 classes - and for a new database:
Use [via](https://github.com/JosephGesnouin/ViaAnnotationTool) to annotate/label the images and follow the template from [that](https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation/blob/master/surgery.py) implementation that takes in consideration two different classes: stick to the JSON template and simply change your classes names, you should be fine. ([git](https://github.com/SUYEgit/Surgery-Robot-Detection-Segmentation) of the dual classes project) 

# how to train just final layers using a previous network trained on coco and a new db with annotated players:
 In [model.py](https://github.com/JosephGesnouin/Mask_RCNN/blob/master/mrcnn/model.py) you have that [train()](https://github.com/JosephGesnouin/Mask_RCNN/blob/master/mrcnn/model.py#L2701-L2794) fonction that trains different layers depending on what params you give. You just have to edit for instance the train call in [football.py](phGesnouin/Mask_RCNN/blob/master/samples/footballplayers/footplayers.py#L227-L247) with the regular expression that fits the amount of layers you want to train. Also a big help for transfer learning comes from the tutorial linked previously and should be sufficient
