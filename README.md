# HABITAT
## Overview
HABITAT stands for **H**igh-resolution **A**rctic **B**uilt **I**nfrastructure and **T**errain **A**nalysis **T**ool. This is a fully-automated, end-to-end geospatial deep learning pipeline designed to map Arctic built infrastructure from &lt;1 m spatial resolution Maxar satellite imagery.

HABITAT model inferencing works in the following sequence:  
(1) Take an entire satellite image and split it into smaller tiles.  
(2) Feed the tiles to a convolutional neural network model trained for infrastructure detection.  
(3) Through semantic segmentation, detect different types of infrastructure in each image tile and predict a segmentation mask.  
(4) Stitch the masks back into a raster with the dimensions of the input satellite image and georeference it.  
(5) Convert the raster into a final georeferenced output map.



This diagram provides a high-level overview of the HABITAT workflow:  
![HABITAT_workflow](https://github.com/PermafrostDiscoveryGateway/HABITAT/assets/77365021/772f455e-19c5-4161-900c-aad2724ac732)


## Environment setup
Install the following major requirements:
```
Name                          Version
python                        3.7
albumentations                1.2.1 
segmentation-models-pytorch   0.2.1
torch                         1.10.1+cu113             
torchvision                   0.11.2+cu113
tensorflow-gpu                2.4.1
tifffile                      2021.11.2 
natsort                       8.3.1
opencv-python                 4.7.0.72 
rasterio                      1.2.10
geopandas                     0.10.2
numpy                         1.19.5
tqdm                          4.65.0
```

## How to run
### Data setup
Model and data setup for both training and inferencing are controlled by configuration files: (final_model_config.py for training and operational_config.py for inferencing). Within these files, you must specify the ROOT_DIR (root directory) and the WORKER_ROOT, which is the "home base" for your data. WORKER_ROOT should contain INPUT_IMG_DIR and INPUT_MASK_DIR, which hold your training image tiles and training mask tiles, respectively. Specify where you want to save the model weights with WEIGHT_PATH, which should be used later when loading the trained model for inferencing. For example, in final_model_config.py, we have written:

```
class Final_Config(object):

    # Give the configuration a distinct name related to the experiment
    NAME = 'ResNet50-UNet++_allSites_duplicateTanks'

    # Set paths to data

    ROOT_DIR = r'/scratch/bbou/eliasm1'
    # ROOT_DIR = r'D:/infra-master'
    WORKER_ROOT =  ROOT_DIR + r'/data/'

    INPUT_IMG_DIR = WORKER_ROOT + r'/256x256/imgs'
    INPUT_MASK_DIR = WORKER_ROOT + r'/256x256/masks'
    TEST_OUTPUT_DIR = ROOT_DIR + r'/test_output'
    PLOT_DIR = ROOT_DIR + r'/plots/' + NAME 
    WEIGHT_DIR = ROOT_DIR + r'/model_weights/' + NAME
```
INPUT_IMG_DIR and INPUT_MASK_DIR should match eachother, like in the following example:

```   
INPUT_IMG_DIR
└───train
│   │   img_10.TIF
│   │   ...
│
└───val
|    │  img_20.TIF
|    │  ...
|
└───test
|   |  img_40.TIF
|   |  ...


INPUT_MASK_DIR
└───train
│   │   mask_10.TIF
│   │   ...
│
└───val
|    │  mask_20.TIF
|    │  ...
|
└───test
|   |  mask_40.TIF
|   |  ...

```

### Model setup
The final trained model that we use for infrastructure mapping with HABITAT is the ResNet-50-UNet++. However, if you wish to train a completely different model with your own parameters, modify the following section in final_model_config.py:

```
    # Configure model training

    SIZE = 256
    CHANNELS = 3
    CLASSES = 10
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax'

    PREPROCESS = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # UNet++
    MODEL = smp.UnetPlusPlus(encoder_name=ENCODER,
                             encoder_weights=ENCODER_WEIGHTS,
                             in_channels=CHANNELS,
                             classes=CLASSES,
                             activation=ACTIVATION)
    
    LOSS = smp.losses.FocalLoss(mode='multilabel')
    LOSS.__name__ = 'FocalLoss'

    METRICS = [smp.utils.metrics.Fscore(threshold=0.5)]
    OPTIMIZER = torch.optim.Adam([dict(params=MODEL.parameters(), lr=0.0001)])
    DEVICE = 'cuda'
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 1
    EPOCHS = 80

    # Select augmentations
    AUGMENTATIONS = [albu.Transpose(p=0.6),
                     albu.RandomRotate90(p=0.6),
                     albu.HorizontalFlip(p=0.6),
                     albu.VerticalFlip(p=0.6)]
```

### Training a model
After you have chosen your desired parameters in final_model_config.py, train the model, by simply running:
```
python model_train.py
```

You can also execute model training on HPC resources by apppending this command at the end of a job script.

### Inferencing on one satellite image
To use a trained model in operational deployment of the HABITAT mapping pipeline on one satellite image:
```
python full_pipeline.py --image <IMAGE_NAME>
```

### Inferencing on multiple satellite images
To automatically run HABITAT on multiple satellite image on an HPC resource, make sure you have specified in operational_config.py (1) the path that holds the input satellite images, (2) the path that will hold the output polygons maps, and (3) the path that contains the trained model weights (model weights can be found here: https://drive.google.com/drive/folders/1wnSIv_oDZlFMHtophpVCiaKSC97uvqEQ?usp=sharing):
```
class Operational_Config(object):

    # Give the configuration a distinct name related to the experiment
    NAME = 'ResNet50-UNet++_allSites_duplicateTanks'

    # Set paths to data

    ROOT_DIR = r'/scratch/bbou/eliasm1'
    WORKER_ROOT =  ROOT_DIR + r'/data/'

    INPUT_SCENE_DIR = ROOT_DIR + r'/alaska_scenes/prudhoe_bay'
    OUTPUT_DIR = ROOT_DIR + r'/inference_output'
    WEIGHT_DIR = ROOT_DIR + r'/model_weights/' + NAME + '.pth'
```
Then you can use:
```
python run_workflow.py
```

In lines 13-15 of run_workflow.py, you can chose how many satellite images in INPUT_IMG_DIR will be input for inferencing:
```
start = 0
end = 5
selected_files = files[start:end]
```

