import arcpy
import os
import geopandas as gpd

arcpy.env.overwriteOutput = True

def calculate_compactness(geom):
    """
    Calculates the compactness of a polygon.
    Compactness = (4 * pi * Area) / (Perimeter^2).
    """
    if geom.is_empty or geom.length == 0:
        return 0  # Handle edge cases where the perimeter is 0
    area = geom.area
    perimeter = geom.length
    return (4 * 3.14159 * area) / (perimeter ** 2)

def process_shapefiles(input_dir, output_dir_buildings, output_dir_roads, start_file, end_file):
    """
    Processes shapefiles by filtering features, splitting by class,
    regularizing building footprints, and generating road centerlines.

    Parameters:
        input_dir (str): Directory containing the input shapefiles.
        output_dir_buildings (str): Directory to save processed building shapefiles.
        output_dir_roads (str): Directory to save processed road shapefiles.
        start_file (int): The starting index of files to process (1-based).
        end_file (int): The ending index of files to process (inclusive).
    """
    # List all shapefiles in the input directory and sort them
    shapefiles = [f for f in os.listdir(input_dir) if f.endswith('.shp')]
    shapefiles.sort()  # Sort shapefiles in ascending name order

    # Limit the range of files to process
    shapefiles = shapefiles[start_file:end_file]  # Convert 1-based index to 0-based indexing for slicing

    print(f"Processing files {start_file} to {end_file} out of {len(shapefiles)} total files.")

    for i, shapefile in enumerate(shapefiles, start=start_file):
        print(f"Processing file {i}/{end_file}: {shapefile}")
        input_path = os.path.join(input_dir, shapefile)

        try:
            # Use ArcPy for processing
            arcpy.MakeFeatureLayer_management(input_path, "temp_layer")

            # Add a field for Area
            arcpy.AddField_management("temp_layer", "Area", "DOUBLE")

            # Calculate Area
            arcpy.CalculateGeometryAttributes_management("temp_layer", [["Area", "AREA"]])

            # Select and delete features with area <= 10
            print(f"Deleting features with area <= 10 for {shapefile}...")
            arcpy.management.SelectLayerByAttribute("temp_layer", "NEW_SELECTION", 'Area <= 10')
            arcpy.management.DeleteFeatures("temp_layer")

            # Split by class = 1 and class = 2
            print(f"Splitting features by class for {shapefile}...")
            class1_layer = "class1_layer"
            class2_layer = "class2_layer"

            arcpy.SelectLayerByAttribute_management("temp_layer", "NEW_SELECTION", 'class = 1')
            arcpy.MakeFeatureLayer_management("temp_layer", class1_layer)

            arcpy.SelectLayerByAttribute_management("temp_layer", "NEW_SELECTION", 'class = 2')
            arcpy.MakeFeatureLayer_management("temp_layer", class2_layer)

            # Process class = 1 (buildings) with Regularize Building Footprints
            print(f"Regularizing building footprints for {shapefile}...")
            building_output = os.path.join(output_dir_buildings, shapefile)
            arcpy.ddd.RegularizeBuildingFootprint(
                class1_layer, building_output, tolerance=1, method="RIGHT_ANGLES", precision=0.25
            )

            # Perform compactness filtering directly after area calculation
            print(f"Filtering regularized buildings by compactness for {shapefile}...")
            regularized_gdf = gpd.read_file(building_output)
            regularized_gdf['compact'] = regularized_gdf.geometry.apply(calculate_compactness)

            # Apply compactness filtering
            filtered_gdf = regularized_gdf[regularized_gdf['compact'] >= 0.15]  # Adjust threshold as needed

            # Save the filtered buildings directly
            filtered_gdf.to_file(building_output)

            print(f"Saved filtered buildings to {building_output}.")

            # # Process class = 2 (roads) with Polygon to Centerline
            # print(f"Generating centerlines for roads in {shapefile}...")
            # road_output = os.path.join(output_dir_roads, shapefile)
            # arcpy.cartography.CollapseHydroPolygon(class2_layer, road_output)

            # Clean up temporary layers
            print(f"Cleaning up temporary layers...")
            arcpy.Delete_management("temp_layer")
            arcpy.Delete_management(class1_layer)
            arcpy.Delete_management(class2_layer)

            print(f"Finished processing {shapefile}.")

        except Exception as e:
            print(f"An error occurred while processing {shapefile}: {e}")
            arcpy.Delete_management("temp_layer", "in_memory")
            arcpy.Delete_management(class1_layer, "in_memory")
            arcpy.Delete_management(class2_layer, "in_memory")
            continue

# Define input and output directories
input_directory = r"D:\HABITAT\maps\ResNet50-UNet++_512_0.5FTL_0.90A_0.75G_0.5CE_3class\russia"
output_directory_buildings = r"D:\HABITAT\maps\postprocessed\russia\buildings"
output_directory_roads = r"D:\HABITAT\maps\postprocessed\russia\roads"

# Ensure output directories exist
os.makedirs(output_directory_buildings, exist_ok=True)
os.makedirs(output_directory_roads, exist_ok=True)

# File range to process
start_file = 0
end_file = 100

# Call the function
process_shapefiles(input_directory, output_directory_buildings, output_directory_roads, start_file, end_file)
