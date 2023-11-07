import tifffile as tiff
import numpy as np
from operational_config import *
from dataloader import *
import os
import rasterio
from rasterio import features
import geopandas as gpd

def stitch_preds(input_img_name, predictions, skipped_indices):

    # Path to the input GeoTIFF satellite image
    input_img_path = os.path.join(Operational_Config.INPUT_SCENE_DIR, input_img_name)

    # Load the full image using tifffile
    image = tiff.imread(input_img_path)

    # Initialize an empty map with the same dimensions as the original satellite image
    final_map = np.zeros_like(image[:, :, 0])

    # Initialize a counter to keep track of the current prediction
    prediction_counter = 0

    #  Tile size in pixels
    tile_size = Operational_Config.SIZE

    # Iterate through the tiles and predictions
    num_rows = image.shape[0] // tile_size
    num_cols = image.shape[1] // tile_size

    for row in range(num_rows):
        for col in range(num_cols):
            top = row * tile_size
            bottom = top + tile_size
            left = col * tile_size
            right = left + tile_size

            # Check if the current tile is in the list of skipped indices
            if (row * num_cols + col) in skipped_indices:
                continue  # Skip this tile

            # Get the prediction for the current tile
            prediction = predictions[prediction_counter]
            prediction = (prediction.squeeze().cpu().numpy().round())
            prediction = np.moveaxis(prediction, 0, 2)
            final_pred = np.argmax(prediction, axis=2)

            # Place the prediction into the final map at the correct position
            final_map[top:bottom, left:right] = final_pred

            # Increment the prediction counter
            prediction_counter += 1

    # Get filename of input image to save new output
    new_file_name = (input_img_name).split('.tif')[0]

    # Path to the new raster of stitched predictions
    stitched_map_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_stitched.tif"%new_file_name)

    # Save the stitched prediction raster as a GeoTIFF image
    tiff.imsave(stitched_map_path, final_map)


def georeference(input_img_name):

    # Path to the input GeoTIFF satellite image
    input_img_path = os.path.join(Operational_Config.INPUT_SCENE_DIR, input_img_name)

    # Define path to save georeferenced raster
    # Get filename of input image to save new output
    new_file_name = (input_img_name).split('.tif')[0]

    # Path to the new raster of stitched predictions
    stitched_map_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_stitched.tif"%new_file_name)

    # Path to the new raster of stitched predictions
    georeferenced_map_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_georef.tif"%new_file_name)

    # Open the source raster to get its CRS and extent
    with rasterio.open(input_img_path) as src_source:
        source_crs = src_source.crs
        source_transform = src_source.transform

        # Open the destination raster to get its profile and data
        with rasterio.open(stitched_map_path) as src_destination:
            dst_profile = src_destination.profile
            destination_data = src_destination.read()

            # Create a new raster with the profile and data from the destination raster
            with rasterio.open(georeferenced_map_path, 'w', **dst_profile) as dst_new:
                # Set the CRS of the new raster to match the source
                dst_new.crs = source_crs

                # Update the transformation matrix to match the extent of the source
                dst_new.transform = source_transform

                # Write the data from the destination raster to the new raster
                dst_new.write(destination_data)



def polygonize(input_img_name):
    # Get filename of input image to save new output
    new_file_name = os.path.splitext(input_img_name)[0]

    # Path of georeferenced raster
    georeferenced_map_path = os.path.join(Operational_Config.OUTPUT_DIR, f"{new_file_name}_georef.tif")

    # Output shapefile path
    output_shapefile_path = os.path.join(Operational_Config.OUTPUT_DIR, f"{new_file_name}.shp")

    # read the raster and polygonize
    with rasterio.open(georeferenced_map_path) as src:
        image = src.read(1, out_dtype='uint16') 
        # Make a mask!
        mask = image != 0
        
    # `results` contains a tuple. With each element in the tuple representing a dictionary 
    # containing the feature (polygon) and its associated raster value
    results = ( {'properties': {'class': int(v)}, 'geometry': s} 
                for (s, v) in (features.shapes(image, mask=mask, transform=src.transform)))
    
    in_shp = gpd.GeoDataFrame.from_features(results).set_crs(crs=src.crs)
    
    # Save the GeoDataFrame to a shapefile
    in_shp.to_file(output_shapefile_path)

# This function will delete intermediate output (stitched raster and georeferenced stiched raster) 
# that we don't need after the workflow is completed for one image scene
def cleanup(input_img_name):

    # Get filename of input image
    new_file_name = os.path.splitext(input_img_name)[0]

    # Path to the new raster of stitched predictions
    stitched_map_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_stitched.tif"%new_file_name)    
    # Delete stitched raster
    os.remove(stitched_map_path)

    # Path of georeferenced raster
    georeferenced_map_path = os.path.join(Operational_Config.OUTPUT_DIR, f"{new_file_name}_georef.tif")
    # Delete georeferenced raster
    os.remove(georeferenced_map_path)