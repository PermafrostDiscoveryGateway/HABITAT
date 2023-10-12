from final_model_config import *
import tensorflow as tf
from torch.utils.data import Dataset as BaseDataset
import os, numpy as np, cv2
import tifffile as tiff
from natsort import natsorted
from typing import Iterator, Optional
from operator import itemgetter

from torch.utils.data import DistributedSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, Sampler


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
        img = img[:,:,1:4]

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
        
        # Remove white pixels
        mask[mask==255] = 0
        
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
        # Extract B, G, R bands, leaving out NIR.
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



# # Dataset classes for multi-GPU code that has not been implemented yet

# # Required class for DistributedSamplerWrapper
# class DatasetFromSampler(Dataset):
#     """Dataset to create indexes from `Sampler`.
#     Args:
#         sampler: PyTorch sampler
#     """

#     def __init__(self, sampler: Sampler):
#         """Initialisation for DatasetFromSampler."""
#         self.sampler = sampler
#         self.sampler_list = None

#     def __getitem__(self, index: int):
#         """Gets element of the dataset.
#         Args:
#             index: index of the element in the dataset
#         Returns:
#             Single element by index
#         """
#         if self.sampler_list is None:
#             self.sampler_list = list(self.sampler)
#         return self.sampler_list[index]

#     def __len__(self) -> int:
#         """
#         Returns:
#             int: length of the dataset
#         """
#         return len(self.sampler)


# # Sampler for distributed data parallel 
# class DistributedSamplerWrapper(DistributedSampler):
#     """
#     Wrapper over `Sampler` for distributed training.
#     Allows you to use any sampler in distributed mode.
#     It is especially useful in conjunction with
#     `torch.nn.parallel.DistributedDataParallel`. In such case, each
#     process can pass a DistributedSamplerWrapper instance as a DataLoader
#     sampler, and load a subset of subsampled data of the original dataset
#     that is exclusive to it.
#     .. note::
#         Sampler is assumed to be of constant size.
#     """

#     def __init__(
#         self,
#         sampler,
#         num_replicas: Optional[int] = None,
#         rank: Optional[int] = None,
#         shuffle: bool = True,
#     ):
#         """
#         Args:
#             sampler: Sampler used for subsampling
#             num_replicas (int, optional): Number of processes participating in
#                 distributed training
#             rank (int, optional): Rank of the current process
#                 within ``num_replicas``
#             shuffle (bool, optional): If true (default),
#                 sampler will shuffle the indices
#         """
#         super(DistributedSamplerWrapper, self).__init__(
#             DatasetFromSampler(sampler),
#             num_replicas=num_replicas,
#             rank=rank,
#             shuffle=shuffle,
#         )
#         self.sampler = sampler

#     def __iter__(self) -> Iterator[int]:
#         """Iterate over sampler.
#         Returns:
#             python iterator
#         """
#         self.dataset = DatasetFromSampler(self.sampler)
#         indexes_of_indexes = super().__iter__()
#         subsampler_indexes = self.dataset
#         return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))