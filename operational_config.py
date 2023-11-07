import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import albumentations as albu


class Operational_Config(object):

    # Give the configuration a distinct name related to the experiment
    NAME = 'ResNet50-UNet++_allSites_duplicateTanks'

    # Set paths to data

    ROOT_DIR = r'/scratch/bbou/eliasm1'
    WORKER_ROOT =  ROOT_DIR + r'/data/'

    INPUT_SCENE_DIR = ROOT_DIR + r'/alaska_scenes/prudhoe_bay'
    OUTPUT_DIR = ROOT_DIR + r'/inference_output'
    WEIGHT_DIR = ROOT_DIR + r'/model_weights/' + NAME + '.pth'

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

