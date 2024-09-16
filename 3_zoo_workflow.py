import os
from local_coastseg.coastseg import coastseg_logs
from local_coastseg.coastseg import zoo_model
from local_coastseg.coastseg.tide_correction import compute_tidal_corrections
from local_coastseg.coastseg import file_utilities

# The Zoo Model is a machine learning model that can be used to extract shorelines from satellite imagery.
# This script will only run a single ROI at a time. If you want to run multiple ROIs, you will need to run this script multiple times.
case = 1
if case == 0:
    # point hope
    sample_direc = "/Users/rdchlkrd/Desktop/krd/coastseg/CoastSeg/data/ID_rtn1_datetime09-04-24__11_29_40/jpg_files/preprocessed_s1_4band"
    model_session_name = "pointhope_4band_2class_combined"
    transects_path = "/Users/rdchlkrd/Desktop/krd/CoastSat.Arctic/data/point_hope_north_2018-07-01_2018-08-01/transects.geojson" # path to the transects geojson file (optional, default will be loaded if not provided)
    # shoreline_path = "/Users/rdchlkrd/Desktop/krd/coastseg/CoastSeg/data/ID_rtn1_datetime09-04-24__11_29_40/shoreline.geojson"
    shoreline_path = ""
if case == 1:
    sample_direc = "/Users/rdchlkrd/Desktop/krd/coastseg/CoastSeg/data/ID_ikf1_datetime09-03-24__12_22_39/jpg_files/preprocessed_s1_4band"
    model_session_name = "dylan_4band_2class_combined"
    transects_path = "/Users/rdchlkrd/Desktop/krd/CoastSat.Arctic/data/dylan_site_2018-01-01_2020-01-01/transects.geojson" # path to the transects geojson file (optional, default will be loaded if not provided)
    # shoreline_path = "/Users/rdchlkrd/Desktop/krd/coastseg/CoastSeg/data/ID_rtn1_datetime09-04-24__11_29_40/shoreline.geojson"
    shoreline_path = "/Users/rdchlkrd/Desktop/krd/coastseg/CoastSeg/data/ID_ikf1_datetime09-03-24__12_22_39/shoreline.geojson"
# Extract Shoreline Settings
settings ={
    'min_length_sl': 500,       # minimum length (m) of shoreline perimeter to be valid
    'max_dist_ref':300,         # maximum distance (m) from reference shoreline to search for valid shorelines. This detrmines the width of the buffer around the reference shoreline  
    'cloud_thresh': 0.5,        # threshold on maximum cloud cover (0-1). If the cloud cover is above this threshold, no shorelines will be extracted from that image
    'dist_clouds': 300,         # distance(m) around clouds where shoreline will not be mapped
    'min_beach_area': 500,      # minimum area (m^2) for an object to be labelled as a beach
    'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    "apply_cloud_mask": True,   # apply cloud mask to the imagery. If False, the cloud mask will not be applied.
}

# The model can be run using the following settings:
model_setting = {
            # "sample_direc": "/Users/rdchlkrd/Desktop/krd/coastseg/CoastSeg/data/ID_vqe1_datetime08-30-24__10_57_35/jpg_files/good", # directory of jpgs  ex. C:/Users/username/CoastSeg/data/ID_lla12_datetime11-07-23__08_14_11/jpg_files/preprocessed/RGB/",
            # "sample_direc": "/Users/rdchlkrd/Desktop/krd/coastseg/CoastSeg/data/ID_urt1_datetime08-31-24__02_32_19/jpg_files/preprocessed_s1_4band", # directory of jpgs  ex. C:/Users/username/CoastSeg/data/ID_lla12_datetime11-07-23__08_14_11/jpg_files/preprocessed/RGB/",
            # "sample_direc": "/Users/rdchlkrd/Desktop/krd/coastseg/CoastSeg/data/ID_ikf1_datetime09-03-24__12_22_39/jpg_files/preprocessed_s1", # directory of jpgs  ex. C:/Users/username/CoastSeg/data/ID_lla12_datetime11-07-23__08_14_11/jpg_files/preprocessed/RGB/",
            "sample_direc": sample_direc,
            "use_GPU": "0",  # 0 or 1 0 means no GPU
            "implementation": "BEST",  # BEST or ENSEMBLE 
            "model_type": "s1_4band_2class_combined", # model name ex. segformer_RGB_4class_8190958
            # "model_type": "s1_sar_4band_2class_ice", # model name ex. segformer_RGB_4class_8190958
            "otsu": False, # Otsu Thresholding
            "tta": False,  # Test Time Augmentation
        }
# Available models can run input "RGB" # or "MNDWI" or "NDWI"
img_type = "SAR_4BAND"
# percentage of no data allowed in the image eg. 0.75 means 75% of the image can be no data
percent_no_data = 0.75

# 1. Set the User configuration Settings
# ---------------------------
# a. ENTER THE NAME OF THE SESSION TO SAVE THE MODEL PREDICTIONS TO
# model_session_name = "duck_2class_ice_30"
# model_session_name = "dylan_site_2class_ice_30"
# model_session_name = "drew_4band_2class_combined"
# model_session_name = "dylan_4band_2class_combined"
# b. ENTER THE DIRECTORY WHERE THE INPUT IMAGES ARE STORED
# -  Example of the directory where the input images are stored ( this should be the /data folder in the CoastSeg directory)
sample_directory = model_setting["sample_direc"] #"C:\development\doodleverse\coastseg\CoastSeg\data\ID_wra5_datetime03-04-24__03_43_01\jpg_files\preprocessed\RGB"


# 2. Save the settings to the model instance 
# -----------------
# Create an instance of the zoo model to run the model predictions
zoo_model_instance = zoo_model.Zoo_Model()
zoo_model_instance.local_model = True
# Set the model settings to read the input images from the sample directory
model_setting["sample_direc"] = sample_directory
model_setting["img_type"] = img_type

# save settings to the zoo model instance
settings.update(model_setting)
# save the settings to the model instance
zoo_model_instance.set_settings(**settings)


# OPTIONAL: If you have a transects and shoreline file, you can extract shorelines from the zoo model outputs
# transects_path = "/Users/rdchlkrd/Desktop/krd/CoastSat.Arctic/data/drew_point_icce_2018-01-01_2019-01-01/transects.geojson" # path to the transects geojson file (optional, default will be loaded if not provided)
# transects_path = "/Users/rdchlkrd/Desktop/krd/CoastSat.Arctic/data/dylan_site_2018-01-01_2020-01-01/transects.geojson" # path to the transects geojson file (optional, default will be loaded if not provided)
# shoreline_path = "/Users/rdchlkrd/Desktop/krd/CoastSat.Arctic/data/dylan_site_2020-01-01_2024-01-01/dylan_site_2020-01-01_2024-01-01_reference_shoreline.geojson" # path to the shoreline geojson file (optional, default will be loaded if not provided)
shoreline_extraction_area_path= "" # path to the shoreline extraction area geojson file (optional)

# 3. Run the model and extract shorelines
# -------------------------------------
zoo_model_instance.run_model_and_extract_shorelines(
            model_setting["sample_direc"],
            session_name=model_session_name,
            shoreline_path=shoreline_path,
            transects_path=transects_path,
            shoreline_extraction_area_path = shoreline_extraction_area_path
        )

# 4. OPTIONAL: Run Tide Correction
# ------------------------------------------
# Tide Correction (optional)
# Before running this snippet, you must download the tide model to the CoastSeg/tide_model folder
# Tutorial: https://github.com/Doodleverse/CoastSeg/wiki/09.-How-to-Download-and-clip-Tide-Model
#  You will need to uncomment the line below to run the tide correction

beach_slope = 0.02 # Slope of the beach (m)
reference_elevation = 0 # Elevation of the beach Mean Sea Level (M.S.L) (m)

# UNCOMMENT THESE LINES TO RUN THE TIDE CORRECTION
# roi_id = file_utilities.get_ROI_ID_from_session(session_name) # read ROI ID from the config.json file found in the extracted shoreline session directory
# compute_tidal_corrections(
#     session_name, [roi_id], beach_slope, reference_elevation
# )