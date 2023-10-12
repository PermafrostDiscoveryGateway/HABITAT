import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import albumentations as albu


class Final_Config(object):

    # Give the configuration a distinct name related to the experiment
    NAME = 'ResNet50-UNet++'

    # Set paths to data

    ROOT_DIR = r'/scratch/bbou/eliasm1'
    # ROOT_DIR = r'D:/infra-master'
    WORKER_ROOT =  ROOT_DIR + r'/data/'

    INPUT_IMG_DIR = WORKER_ROOT + r'/256x256/imgs'
    INPUT_MASK_DIR = WORKER_ROOT + r'/256x256/masks'
    TEST_OUTPUT_DIR = ROOT_DIR + r'/test_output'
    PLOT_PATH = ROOT_DIR + r'/plots/' + NAME + '_allSites_duplicateTanks'
    WEIGHT_PATH = ROOT_DIR + r'/model_weights/' + NAME + '_allSites_duplicateTanks'

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

