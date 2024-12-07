# This script is used to compute the balanced class weights for weighted cross entropy loss.
# We will load in the training mask tiles into an array and flatten the array.

from final_model_config import *
import numpy as np, cv2
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
import tifffile as tiff

y_train_dir = Final_Config.INPUT_MASK_DIR + '/train'

# Iterate through training masks
train_masks = next(os.walk(y_train_dir))[2]

# Define parameters for mask resizing
IMG_SIZE = Final_Config.SIZE
IMG_CHANNELS = Final_Config.CHANNELS

# Create new arrays to store training mask tiles
Y = np.zeros((len(train_masks), IMG_SIZE, IMG_SIZE), dtype=np.uint8)

# Resize training masks
for n, id_ in tqdm(enumerate(train_masks), total=len(train_masks)):
    # Load masks
    mask = tiff.imread(y_train_dir + '/' + id_)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # Ensure only buildings and roads are treated as classes
    mask[mask == 255] = 0
    mask[mask == 2] = 1
    mask[mask == 3] = 1
    mask[mask == 4] = 1
    mask[mask == 5] = 2
    mask[mask == 6] = 0
    mask[mask == 7] = 0
    mask[mask == 8] = 0
    mask[mask == 9] = 1
    mask[mask == 15] = 0

    Y[n] = mask

Y = Y.flatten()

# Count the number of pixels in each class
unique, counts = np.unique(Y, return_counts=True)
class_pixel_counts = dict(zip(unique, counts))

# Print the number of pixels in each class
for class_label, pixel_count in class_pixel_counts.items():
    print(f"Class {class_label}: {pixel_count} pixels")