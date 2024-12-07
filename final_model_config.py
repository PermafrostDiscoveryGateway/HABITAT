import segmentation_models_pytorch as smp
import torch
import albumentations as albu
from segmentation_models_pytorch import utils
from FTL import *

class Final_Config(object):

    # Give the configuration a distinct name related to the experiment
    NAME = 'ResNet101-UNet++_512_NO_FBANK_0.75CE_0.25dice_3class_80epochs'

    # Set paths to data

    ROOT_DIR = r'/scratch/08968/eliasm1/HABITAT'
    WORKER_ROOT =  ROOT_DIR + r'/data/'

    INPUT_IMG_DIR = WORKER_ROOT + r'/512x512/imgs'
    INPUT_MASK_DIR = WORKER_ROOT + r'/512x512/masks'
    TEST_OUTPUT_DIR = ROOT_DIR + r'/test_output/'
    PLOT_DIR = ROOT_DIR + r'/plots/' + NAME 
    WEIGHT_DIR = ROOT_DIR + r'/model_weights/' + NAME

    # Configure model training

    SIZE = 512
    CHANNELS = 3
    CLASSES = 3
    ENCODER = 'resnet101'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax'

    PREPROCESS = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # UNet++
    MODEL = smp.UnetPlusPlus(encoder_name=ENCODER,
                             encoder_weights=ENCODER_WEIGHTS,
                             in_channels=CHANNELS,
                             classes=CLASSES,
                             activation=ACTIVATION)

    # Use Cross-entropy Focal Tverksy loss
    LOSS = FocalTverskyLoss(alpha=0.99, gamma=0.25, weight_ce=1, weight_tversky=0.5)
    LOSS.__name__ = 'FTL'

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


