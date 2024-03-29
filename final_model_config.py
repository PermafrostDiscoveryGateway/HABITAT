import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import albumentations as albu
from segmentation_models_pytorch import utils
from focaldiceloss import FocalDiceLoss
from CE_DiceLoss import CEDiceLoss

class Final_Config(object):

    # Give the configuration a distinct name related to the experiment
    NAME = 'ResNet50-UNet++_allAugs_weighted_CE-Dice_4class'

    # Set paths to data

    ROOT_DIR = r'/scratch/bbou/eliasm1'
    # ROOT_DIR = r'D:/infra-master'
    WORKER_ROOT =  ROOT_DIR + r'/data/'

    INPUT_IMG_DIR = WORKER_ROOT + r'/256x256/imgs'
    INPUT_MASK_DIR = WORKER_ROOT + r'/256x256/masks'
    TEST_OUTPUT_DIR = ROOT_DIR + r'/test_output'
    PLOT_DIR = ROOT_DIR + r'/plots/' + NAME 
    WEIGHT_DIR = ROOT_DIR + r'/model_weights/' + NAME

    # Configure model training

    SIZE = 256
    CHANNELS = 3
    CLASSES = 4
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

    # Use Focal loss
    # LOSS = smp.losses.FocalLoss(mode='multilabel')
    # LOSS.__name__ = 'FocalLoss'

    # Use combined Focal-Dice Loss
    # FOCAL_DICE_LOSS = FocalDiceLoss(focal_weight=0.75, dice_weight=0.25)
    # FOCAL_DICE_LOSS.__name__ = 'FocalDiceLoss'

    # LOSS = FOCAL_DICE_LOSS

    # Use CE-Dice Loss
    LOSS = CEDiceLoss()
    LOSS.__name__ = 'CE_Dice'

    METRICS = [utils.metrics.Fscore(threshold=0.5)]
    OPTIMIZER = torch.optim.Adam([dict(params=MODEL.parameters(), lr=0.0001)])
    DEVICE = 'cuda'
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 1
    EPOCHS = 80

    # Select augmentations
    # AUGMENTATIONS = [albu.Transpose(p=0.6),
    #                  albu.RandomRotate90(p=0.6),
    #                  albu.HorizontalFlip(p=0.6),
    #                  albu.VerticalFlip(p=0.6)]

    AUGMENTATIONS = [albu.MotionBlur(blur_limit=(3,7), p=0.18),
                     albu.CLAHE(p=0.25),
                     albu.GaussNoise(var_limit=(10.0,30.0), per_channel=True, mean=0.0, p=0.18),
                     albu.RGBShift(r_shift_limit=(-13,13), g_shift_limit=(-15,60), b_shift_limit=(-13,13), p=0.18),
                     albu.HueSaturationValue(hue_shift_limit=(-10,10), sat_shift_limit=(-10,10), val_shift_limit=(-10,10), p=0.23),
                     albu.RandomBrightnessContrast(p=0.30),
                     albu.RandomGamma(p=0.15)
                    ] 

