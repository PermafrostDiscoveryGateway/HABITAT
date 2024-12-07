import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import albumentations as albu
from segmentation_models_pytorch import utils


class Operational_Config(object):

    # Give the configuration a distinct name related to the experiment
    NAME = 'ResNet50-UNet++_512_0.5FTL_0.90A_0.75G_0.5CE_3class'

    # Set paths to data

    ROOT_DIR = r'/scratch/08968/eliasm1/HABITAT'
    WORKER_ROOT =  ROOT_DIR + r'/data/'

    INPUT_SCENE_DIR = WORKER_ROOT + r'/russia_scenes'
    OUTPUT_DIR = ROOT_DIR + r'/inference_output/' + NAME + r'/russia'
    WEIGHT_DIR = ROOT_DIR + r'/model_weights/' + NAME + '.pth'
    CLEAN_DATA_DIR = WORKER_ROOT + r'/cleaning_data/'
    FOOTPRINT_DIR = WORKER_ROOT + r'/footprints/' + 'russia_pansh_proj_fp.shp'

    # Configure model

    SIZE = 512
    OVERLAP_FACTOR = 0.5
    CHANNELS = 3
    CLASSES = 3
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


