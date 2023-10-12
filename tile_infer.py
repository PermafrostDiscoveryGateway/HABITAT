import tifffile as tiff
import numpy as np
from operational_config import *
from tqdm import tqdm
from dataloader import *
import os

def tile_image(input_img_name):

    #  Tile size in pixels
    tile_size = Operational_Config.SIZE
    
    # Load the full image using tifffile
    image = tiff.imread(input_img_name)

    # Calculate the number of rows and columns of tiles
    num_rows = image.shape[0] // tile_size
    num_cols = image.shape[1] // tile_size

    tiles = []
    skipped_indices = []  # Initialize a list to store skipped tile indices

    for row in range(num_rows):
        for col in range(num_cols):
            top = row * tile_size
            bottom = top + tile_size
            left = col * tile_size
            right = left + tile_size

            # Extract the tile from the image
            tile = image[top:bottom, left:right, ...]

            # Check if all pixels in the tile are equal to the "no-data" value (65536)
            if not np.all(tile == 0):
                tiles.append(tile)
            else:
                skipped_indices.append(row * num_cols + col)  # Record the skipped tile index

    return tiles, skipped_indices


def infer_image(input_img_name):

    # Path to the input GeoTIFF satellite image
    input_img_path = os.path.join(Operational_Config.INPUT_IMG_DIR, input_img_name)

    # Split the image into tiles
    image_tiles, skipped_indices = tile_image(input_img_path)

    # Create a GeoTIFF dataset with the list of image tiles
    dataset = InferDataset(image_tiles, preprocessing=get_preprocessing_test(Operational_Config.PREPROCESS))

    # Load the best saved checkpoint
    best_model = torch.load(Operational_Config.WEIGHT_PATH)

    # Move the model to the GPU
    best_model = best_model.to('cuda')

    # Set the model to evaluation mode
    best_model.eval()

    # Create an empty list to store predictions
    predictions = []

    # Perform inference on tiles
    for i, tile in tqdm(enumerate(dataset), total=len(dataset)):
        # Keep the tile data on the CPU
        tile = tile.astype(np.float32)
        tile = to_tensor(tile)

        # Transfer the tile data to the GPU for prediction
        # Transpose the tensor to [1, 3, 256, 256]
        tile = tile.transpose(2, 0, 1)
        tile = torch.from_numpy(tile).unsqueeze(0).to('cuda')
        tile = tile.permute(0, 3, 1, 2)  # Transpose to [1, 3, 256, 256]

        with torch.no_grad():
            prediction = best_model(tile)

        # Append the prediction to the list
        predictions.append(prediction.cpu())
        # Delete the input tile from GPU memory
        del tile
        torch.cuda.empty_cache()

    return predictions, skipped_indices







