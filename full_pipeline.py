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

# Perform tiling of input satellite image scene and infrastructure detection through model inferencing
print("Satellite image being split and tiles are being fed to infrastructure detection model for inferencing...")
predictions, skipped_indices = infer_image(image_name)
print("Tiling and inferencing complete.")

# Stitch predictions into output raster map
print("Tile predictions being stitched together into output raster map")
stitch_preds(image_name, predictions, skipped_indices)
print("Stitching complete.")

# Georeference the stitched map
print("Stitched map is now being georeferenced...")
georeference(image_name)
print("Georeferencing complete.")

# Polygonize the georeferenced raster map
print("Stitched raster map is now being polygonized...")
polygonize(image_name)
print("Polygonization complete.")

# Delete intermediate output
print("Deleting stitched raster and georeferenced stitched raster")
cleanup(image_name)
print("Cleanup complete.")

print("Workflow complete!")
