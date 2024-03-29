from operational_config import *
from postprocess import *
from tile_infer import *
import argparse

parser = argparse.ArgumentParser(
    description='Run infrastructure detection CNN model in inferencing mode.')

parser.add_argument("--image", required=False,
                    metavar="<command>",
                    help="Image name")

args = parser.parse_args()

image_name = args.image
footprint_shp = Operational_Config.FOOTPRINT_DIR

# Only performing clipping if an image footprint shapefile is specified in operational_config.py
if footprint_shp is not None:
    # Clip input satellite image based on master footprint shapefile to remove overlapping areas
    print("Satellite image is being clipped to remove overlapping areas")
    clip_image(image_name, footprint_shp)
    print("Clipping complete.")

# Perform tiling of input satellite image scene and infrastructure detection through model inferencing
print("Satellite image being split and tiles are being fed to infrastructure detection model for inferencing...")
predictions, skipped_indices = infer_image(image_name)
print("Tiling and inferencing complete.")

# Stitch predictions into output raster map
print("Tile predictions being stitched together into output raster map")
stitch_preds(image_name, predictions, skipped_indices)
print("Stitching complete.")

# Perform morphological processing to improve road connectivity
print("Starting morphological processing of road predictions")
morphological_processing(image_name)
print("Morphological processing complete.")

# Georeference the stitched map
print("Stitched map is now being georeferenced...")
georeference(image_name)
print("Georeferencing complete.")

# Polygonize the georeferenced raster map
print("Stitched raster map is now being polygonized...")
polygonize_and_simplify(image_name)
print("Polygonization complete.")

# Delete intermediate output
print("Deleting stitched raster and georeferenced stitched raster")
cleanup(image_name)
print("Cleanup complete.")

print("Workflow complete!")
