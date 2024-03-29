from final_model_config import *
import tensorflow as tf
from torch.utils.data import Dataset as BaseDataset
import os, numpy as np, cv2
import tifffile as tiff
from natsort import natsorted

IMG_SIZE = Final_Config.SIZE
IMG_CHANNELS = Final_Config.CHANNELS
CLASSES = Final_Config.CLASSES

# Create PyTorch dataset class for model training/validation
class Dataset(BaseDataset):
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = natsorted(os.listdir(images_dir))
        self.mask_ids = natsorted(os.listdir(masks_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # Read in TIFF image tile
        img = tiff.imread(self.images_fps[i])
        # Extract Green, Red, NIR channels.
        img = img[:,:,0:3]

        # Apply minimum-maximum normalization.
        img = cv2.normalize(img, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = img.astype(np.uint8)
        G, R, N = cv2.split(img)
        # Equalize histograms
        out_G = cv2.equalizeHist(G)
        out_R = cv2.equalizeHist(R)
        out_N = cv2.equalizeHist(N)
        
        final_img = cv2.merge((out_G, out_R, out_N))
        # Ensure image tiles are 256x256 pixels
        image = cv2.resize(final_img, (IMG_SIZE, IMG_SIZE))
        
        # Read in TIFF mask tile
        mask = tiff.imread(self.masks_fps[i])
        # Ensure image tiles are 256x256 pixels. Interpolation argument must be set to nearest-neighbor
        # to preserve ground truth.
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_NEAREST)

        # 9 classes were used in creation of the training dataset, but I have decided to condense the classification hierarchy 
        # to a single building superclass, tank class and road class. Therefore, the below code reassigns the values in the 
        # NumPy array of the segmentation mask for building and road segmentation only (4 unique values)
        mask[mask==255] = 0
        mask[mask==2] = 1
        mask[mask==3] = 1
        mask[mask==4] = 1
        mask[mask==5] = 2
        mask[mask==6] = 0
        mask[mask==7] = 0
        mask[mask==8] = 0
        mask[mask==9] = 3


        # One-hot encode masks for multi-class segmentation
        # (10 infrastructure classes, or 7 if we merge building classes)
        onehot_mask = tf.one_hot(mask, CLASSES, axis = 0)
        mask = np.stack(onehot_mask, axis=-1).astype('float')
      
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


# Create PyTorch dataset class for model inferencing
class InferDataset(Dataset):
    def __init__(self, 
                 image_tiles, 
                 preprocessing=None
    ):
        self.image_tiles = image_tiles
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_tiles)

    def __getitem__(self, idx):
        img = self.image_tiles[idx]
        # Extract Green, Red, NIR channels.
        img = img[:, :, 1:4]
        # Apply minimum-maximum normalization.
        img = cv2.normalize(img, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = img.astype(np.uint8)
        G, R, N = cv2.split(img)
        # Equalize histograms
        out_G = cv2.equalizeHist(G)
        out_R = cv2.equalizeHist(R)
        out_N = cv2.equalizeHist(N)
        
        final_img = cv2.merge((out_G, out_R, out_N))

        # Ensure image tiles are 256x256 pixels
        image = cv2.resize(final_img, (IMG_SIZE, IMG_SIZE))

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        return image


# Create helper classes for data preprocessing and augmentation.

def get_training_augmentation():
    train_transform = Final_Config.AUGMENTATIONS
    return albu.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_preprocessing_test(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)
