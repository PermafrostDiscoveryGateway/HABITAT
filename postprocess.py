import tifffile as tiff
import numpy as np
from operational_config import *
from dataloader import *
import os
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.features import shapes
from skimage.morphology import disk
from skimage import morphology
from shapely.geometry import shape
from operational_config import *

if not os.path.exists(Operational_Config.OUTPUT_DIR):
    os.mkdir(Operational_Config.OUTPUT_DIR)

# def stitch_preds(input_img_name, predictions, skipped_indices):

#     # Get filename of input image
#     new_file_name = os.path.splitext(input_img_name)[0]

#     # Path to the input GeoTIFF satellite image
#     input_img_path = os.path.join(Operational_Config.INPUT_SCENE_DIR, input_img_name)
#     # Path to clipped input GeoTIFF satellite image
#     clipped_img_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_clipped.tif"%new_file_name)

#     if Operational_Config.FOOTPRINT_DIR is not None:
#         # Load the full image using tifffile
#         image = tiff.imread(clipped_img_path)
#     else: 
#         image = tiff.imread(input_img_path)

#     # Initialize an empty map with the same dimensions as the original satellite image
#     final_map = np.zeros_like(image[:, :, 0])

#     # Initialize a counter to keep track of the current prediction
#     prediction_counter = 0

#     #  Tile size in pixels
#     tile_size = Operational_Config.SIZE

#     # Iterate through the tiles and predictions
#     num_rows = image.shape[0] // tile_size
#     num_cols = image.shape[1] // tile_size

#     for row in range(num_rows):
#         for col in range(num_cols):
#             top = row * tile_size
#             bottom = top + tile_size
#             left = col * tile_size
#             right = left + tile_size

#             # Check if the current tile is in the list of skipped indices
#             if (row * num_cols + col) in skipped_indices:
#                 continue  # Skip this tile

#             # Get the prediction for the current tile
#             prediction = predictions[prediction_counter]
#             prediction = (prediction.squeeze().round())
#             prediction = np.moveaxis(prediction, 0, 2)
#             final_pred = np.argmax(prediction, axis=2)

#             # The following code block that is commented out is for building boundary regularization. For now, I am commenting it out
#             # since I have found a more superior method, which is the Regularize Building Footprints tool in ArcGIS Pro, a proprietary software. 
#             # If you have access to ArcGIS Pro, I suggest that you run this detection pipeline, then take the output maps and run boundary regularization
#             # on buildings there.
            
#             # final_pred = final_pred.astype(np.uint8)

#             # # Create a new array to store the output data
#             # output_data = final_pred.astype(np.uint8).copy()

#             # # Only apply on cells with detected buildings
#             # building_cells = final_pred == 1
#             # building_cells = building_cells.astype(np.uint8)

#             # # Denoising
#             # ori_img = cv2.medianBlur(building_cells, 5)
#             # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#             # ori_img = cv2.dilate(ori_img, kernel, iterations=1)

#             # # Connected component analysis
#             # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_img, connectivity=8)

#             # output_data[output_data==1] = 0

#             # for i in range(1, num_labels):
#             #     img = np.zeros_like(labels)
#             #     index = np.where(labels==i)
#             #     img[index] = 255
#             #     img = np.array(img, dtype=np.uint8)

#             #     regularization_contour = boundary_regularization(img).astype(np.int32)

#             #     cv2.fillPoly(img=output_data, pts=[regularization_contour], color=1)

#             # Place the prediction into the final map at the correct position
#             final_map[top:bottom, left:right] = final_pred    #  output_data instead of final_pred if using the boundary regularization code

#             # Increment the prediction counter
#             prediction_counter += 1

#     # Path to the new raster of stitched predictions
#     stitched_map_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_stitched.tif"%new_file_name)

#     # Save the stitched prediction raster as a GeoTIFF image
#     tiff.imsave(stitched_map_path, final_map)
    
#     # Path to the new raster of stitched predictions
#     stitched_map_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_stitched.tif"%new_file_name)

#     # Save the stitched prediction raster as a GeoTIFF image
#     tiff.imsave(stitched_map_path, final_map)

def stitch_preds(input_img_name, predictions, skipped_indices, no_data_masks):
    # Get filename of input image
    new_file_name = os.path.splitext(input_img_name)[0]

    # Path to the input GeoTIFF satellite image
    input_img_path = os.path.join(Operational_Config.INPUT_SCENE_DIR, input_img_name)
    # Path to clipped input GeoTIFF satellite image
    clipped_img_path = os.path.join(Operational_Config.OUTPUT_DIR, "%s_clipped.tif" % new_file_name)

    if Operational_Config.FOOTPRINT_DIR is not None:
        # Load the full image using tifffile
        image = tiff.imread(clipped_img_path)
    else:
        image = tiff.imread(input_img_path)

    # Initialize an empty map with the same dimensions as the original satellite image
    final_map = np.zeros_like(image[:, :, 0], dtype=np.uint16)
    count_map = np.zeros_like(image[:, :, 0], dtype=np.uint16)  # To track how many times each pixel is included

    # Initialize a counter to keep track of the current prediction
    prediction_counter = 0

    # Tile size in pixels
    tile_size = Operational_Config.SIZE
    overlap = Operational_Config.OVERLAP_FACTOR  # Define your overlap factor (e.g., 0.2)
    stride = int(tile_size * (1 - overlap))  # Calculate stride based on overlap

    # Calculate the number of rows and columns considering overlap
    num_rows = (image.shape[0] - tile_size) // stride + 1
    num_cols = (image.shape[1] - tile_size) // stride + 1

    # Iterate through the tiles and predictions
    for row in range(num_rows):
        for col in range(num_cols):
            top = row * stride
            bottom = top + tile_size
            left = col * stride
            right = left + tile_size

            # Skip this tile if it has been skipped
            if (row * num_cols + col) in skipped_indices:
                continue

            # Get the prediction for the current tile
            prediction = predictions[prediction_counter].squeeze().round()

            # Convert the prediction to a numpy array if it's a torch tensor
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.numpy()

            prediction = np.moveaxis(prediction, 0, 2)  # Moves axis 0 to position 2

            # Get the NoData mask for the current tile
            no_data_mask_tile = no_data_masks[row * num_cols + col]

            final_pred = np.argmax(prediction, axis=2).astype(np.uint16)  # Cast to uint16

            no_data_mask_tile = no_data_mask_tile.squeeze()  # Ensure NoData mask is 2D
            
            # Apply NoData mask by setting the prediction at NoData pixels to 0
            final_pred[no_data_mask_tile] = 0  # Set NoData pixels to 0

            # Combine predictions into final map using the maximum value across overlaps
            final_map[top:bottom, left:right] = np.maximum(final_map[top:bottom, left:right], final_pred)

            # Increment the prediction counter
            prediction_counter += 1

    # Save the final map without averaging
    stitched_map_path = os.path.join(Operational_Config.OUTPUT_DIR, "%s_stitched.tif" % new_file_name)
    tiff.imsave(stitched_map_path, final_map.astype(np.uint8))  # Ensure proper dtype for saving





def morphological_processing(input_img_name):
    # Get filename of input image to save new output
    new_file_name = os.path.splitext(input_img_name)[0]

    # Path to the raster of stitched predictions
    stitched_map_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_stitched.tif"%new_file_name)

    # Path to the new morphologically processed raster
    morph_process_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_morph.tif"%new_file_name)

    # read the raster
    with rasterio.open(stitched_map_path) as src:
        image = src.read(1, out_dtype='uint16') 

    print("Applying dilation to road cells...")
    # Create a new array to store the output data
    output_data = image.copy()

    # Apply morphological processing only to cells with value 2
    road_cells = image == 2

    # Number of iterations for dilation
    num_iterations = 5

    # Apply dilation for each iteration
    for i in range(num_iterations):
        print(f"Iteration {i+1} of dilation started...")
        dilated_road_cells = morphology.binary_dilation(road_cells, selem=disk(5))
        print(f"Iteration {i+1} of dilation completed.")

    # Set road cells to 0
    output_data[road_cells] = 0

    # Add dilated road cells to output data
    output_data[dilated_road_cells] = 2

    print("Saving new raster data with dilated roads cells...")
    # Save the new version of the prediction raster wtih dilated roads
    tiff.imwrite(morph_process_path, output_data.astype(image.dtype))

    print("Processing completed successfully.")


def georeference(input_img_name):

    # Get filename of input image
    new_file_name = os.path.splitext(input_img_name)[0]

    if Operational_Config.FOOTPRINT_DIR is not None:
        # Path to clipped input GeoTIFF satellite image
        source_img_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_clipped.tif"%new_file_name)
    else: 
        # Path to the input GeoTIFF satellite image
        source_img_path = os.path.join(Operational_Config.INPUT_SCENE_DIR, input_img_name)


    # Path to the new morphologically processed raster
    morph_process_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_morph.tif"%new_file_name)

    # Path to the new raster of stitched predictions
    georeferenced_map_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_georef.tif"%new_file_name)

    with rasterio.Env(CHECK_DISK_FREE_SPACE="NO"):
        # Open the source raster to get its CRS and extent
        with rasterio.open(source_img_path) as src_source:
            source_crs = src_source.crs
            source_transform = src_source.transform

            # Open the destination raster to get its profile and data
            with rasterio.open(morph_process_path) as src_destination:
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
                    

def simplify_polygon(polygon, tolerance=0.5):
    # Convert the shapely geometry to a GeoSeries
    geo_series = gpd.GeoSeries([polygon])

    # Use the simplify method to simplify the geometry
    simplified_geometry = geo_series.simplify(tolerance=tolerance)

    # Extract the simplified polygon from the GeoSeries
    simplified_polygon = simplified_geometry.iloc[0]

    return simplified_polygon

def clean_predictions(polygons, classes):
    # Path to data that will be used to remove false positives
    NLCD_path = Operational_Config.CLEAN_DATA_DIR + 'NLCD_developed_poly_AK_project.shp'

    # Create a GeoDataFrame from the list of Shapely geometries and class values
    infra_data = {'geometry': polygons, 'class': classes}
    infra_polygons = gpd.GeoDataFrame(infra_data, crs="EPSG:3413")

    # Get total bounds for clipping
    infra_total_bounds = infra_polygons.total_bounds

    # Create a Polygon geometry from the bounding box coordinates
    bbox_polygon = Polygon([(infra_total_bounds[0], infra_total_bounds[1]),
                            (infra_total_bounds[0], infra_total_bounds[3]),
                            (infra_total_bounds[2], infra_total_bounds[3]),
                            (infra_total_bounds[2], infra_total_bounds[1])])
    
    # Create a GeoDataFrame with the bounding box geometry
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon])
    # Set the CRS of the GeoDataFrame
    bbox_gdf.crs = infra_polygons.crs

    # Load the shapefile with polygons to retain
    NLCD_polygons = gpd.read_file(NLCD_path, bbox=bbox_gdf)

    # Buffer the NLCD polygons by 30 meters
    buffered_NLCD = NLCD_polygons.buffer(30)

    # Clip the infrastructure polygons using the union of NLCD polygons + 30 m buffer
    cleaned_polygons = gpd.clip(infra_polygons, buffered_NLCD)

    return cleaned_polygons

def polygonize_and_simplify(input_img_name):
    # Get filename of input image to save new output
    new_file_name = os.path.splitext(input_img_name)[0]

    # Path of georeferenced raster
    georeferenced_map_path = os.path.join(Operational_Config.OUTPUT_DIR, f"{new_file_name}_georef.tif")

    # Output Shapefile path
    # simpli_shapefile_path = os.path.join(Operational_Config.OUTPUT_DIR, f"{new_file_name}_simplified.shp")
    final_shapefile_path = os.path.join(Operational_Config.OUTPUT_DIR, f"{new_file_name}_final.shp")

    # Read the raster and polygonize
    with rasterio.open(georeferenced_map_path) as src:
        image = src.read(1, out_dtype='uint16') 
        # Make a mask!
        mask = image != 0

        # `results` contains a tuple. Each element in the tuple represents a dictionary 
        # containing the feature (polygon) and its associated raster value
        results = [{'properties': {'class': int(v)}, 'geometry': s} 
                   for (s, v) in shapes(image, mask=mask, transform=src.transform)]

    # Extract geometries from results
    polygons = [shape(result['geometry']) for result in results]
    # Extract class values from results
    classes = [result['properties']['class'] for result in results]

    # # Simplify each polygon using the geopandas simplify method
    # simplified_polygons = [simplify_polygon(poly) for poly in polygons]

    # # Clean the data based on specified shapefile
    # cleaned_polygons = clean_predictions(simplified_polygons, classes)

    with rasterio.open(georeferenced_map_path) as src_source:
        source_crs = src_source.crs
        source_transform = src_source.transform

    # Create a GeoDataFrame from the list of Shapely geometries and class values
    infra_data = {'geometry': polygons, 'class': classes}
    infra_polygons = gpd.GeoDataFrame(infra_data, crs=source_crs)

    # Save the cleaned GeoDataFrame to a Shapefile
    infra_polygons.to_file(final_shapefile_path)

        
# This function will delete intermediate output (stitched raster and georeferenced stiched raster) 
# that we don't need after the workflow is completed for one image scene
def cleanup(input_img_name):

    # Get filename of input image
    new_file_name = os.path.splitext(input_img_name)[0]

    # Path to the new raster of stitched predictions
    stitched_map_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_stitched.tif"%new_file_name)    
    # Delete stitched raster
    os.remove(stitched_map_path)

    # Path to the new morphologically processed raster
    morph_process_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_morph.tif"%new_file_name)
    # Delete
    os.remove(morph_process_path)

    # Path of georeferenced raster
    georeferenced_map_path = os.path.join(Operational_Config.OUTPUT_DIR, f"{new_file_name}_georef.tif")
    # Delete georeferenced raster
    os.remove(georeferenced_map_path)

    if Operational_Config.FOOTPRINT_DIR is not None:
        # Path to clipped raster
        clipped_img_path = os.path.join(Operational_Config.OUTPUT_DIR,"%s_clipped.tif"%new_file_name)
        # Delete clipped raster
        os.remove(clipped_img_path)
