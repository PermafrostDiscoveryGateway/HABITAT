import tifffile as tiff
import numpy as np
from operational_config import *
from tqdm import tqdm
from dataloader import *
import os
from utils_segformer import (
    draw_segmentation_map, 
    image_overlay,
    predict
)
import rasterio
from rasterio.mask import mask
import geopandas as gpd

# from transformers import (
#     SegformerFeatureExtractor, 
#     SegformerForSemanticSegmentation
# )

from config_segformer import VIS_LABEL_MAP as LABEL_COLORS_LIST

def clip_image(input_img_name, footprint_shp):

    # Path to the input GeoTIFF satellite image
    input_img_path = os.path.join(Operational_Config.INPUT_SCENE_DIR, input_img_name)

    # Get filename of input image to save new output
    new_file_name = os.path.splitext(input_img_name)[0]

    # Path at which clipped raster will be saved
    clipped_img_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_clipped.tif"%new_file_name)

    # Read the footprint shapefile
    footprints = gpd.read_file(footprint_shp)

    # Filter footprints by filename
    footprint = footprints[footprints['O_FILENAME'] == os.path.basename(input_img_name)]

    # If there are no matching footprints, return empty lists
    if len(footprint) == 0:
        return [], []

    # Get the geometry of the footprint
    footprint_geom = footprint.geometry.values[0]

    # Open the image using rasterio
    with rasterio.open(input_img_path) as src:
        # Read the image data
        out_image, out_transform = rasterio.mask.mask(src, [footprint_geom], crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    # Save clipped image
    with rasterio.Env(CHECK_DISK_FREE_SPACE="NO"):
        with rasterio.open(clipped_img_path, "w", **out_meta) as dest:
            dest.write(out_image)

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

    # Get filename of input image
    new_file_name = os.path.splitext(input_img_name)[0]

    # Path to clipped input GeoTIFF satellite image
    clipped_img_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_clipped.tif"%new_file_name)

    # Path to the input GeoTIFF satellite image
    input_img_path = os.path.join(Operational_Config.INPUT_SCENE_DIR, input_img_name)

    if Operational_Config.FOOTPRINT_DIR is not None:
        # Split the image into tiles
        image_tiles, skipped_indices = tile_image(clipped_img_path)
    else:
         image_tiles, skipped_indices = tile_image(input_img_path) 

    # Create a GeoTIFF dataset with the list of image tiles
    dataset = InferDataset(image_tiles, preprocessing=get_preprocessing_test(Operational_Config.PREPROCESS))

    # Load the best saved checkpoint
    best_model = torch.load(Operational_Config.WEIGHT_DIR)

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


# def infer_image_segformer(input_img_name):

#     # Path to the input GeoTIFF satellite image
#     input_img_path = os.path.join(Operational_Config.INPUT_SCENE_DIR, input_img_name)

#     # Split the image into tiles
#     image_tiles, skipped_indices = tile_image(input_img_path)

#     # Create a GeoTIFF dataset with the list of image tiles
#     dataset = InferDatasetSegFormer(image_tiles)

#     extractor = SegformerFeatureExtractor()
#     model = SegformerForSemanticSegmentation.from_pretrained(Operational_Config.SEGFORMER_WEIGHTS_DIR)
#     model.to('cuda').eval()

#     # Create an empty list to store predictions
#     predictions = []

#     # Perform inference on tiles
#     for i, tile in tqdm(enumerate(dataset), total=len(dataset)):
#         # Keep the tile data on the CPU
#         tile = tile.astype(np.uint8)
#         # tile = to_tensor(tile)

#         # Transfer the tile data to the GPU for prediction
#         # Transpose the tensor to [1, 3, 256, 256]
#         # tile = tile.transpose(2, 0, 1)
#         # tile = torch.from_numpy(tile).unsqueeze(0).to('cuda')
#         # tile = tile.permute(0, 3, 1, 2)  # Transpose to [1, 3, 256, 256]

#         # Get labels.
#         labels = predict(model, extractor, tile, 'cuda')
        
#         # Get segmentation map.
#         # seg_map = draw_segmentation_map(
#         #     labels.cpu(), LABEL_COLORS_LIST
#         # )
#         # Append the prediction to the list
#         predictions.append(labels.cpu())
#         # Delete the input tile from GPU memory
#         del tile
#         torch.cuda.empty_cache()

#     return predictions, skipped_indices








