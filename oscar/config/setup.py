import os, yaml

here = os.path.dirname(__file__)
config_path = os.path.join(here, "io_config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)


#### FROM CONFIG FILE ############
OVERWRITE_DOWNLOAD = config['general']['overwrite_download']
OVERWRITE_CURRENT = config['general']['overwrite_current']
PLOT_CURRENTS = config['general']['plot_currents']
DO_VALIDATION = config['general']['do_validation']
SAVE_COLOCATION = config['general']['do_save_colocation']
DO_CHECKS = config['general']['do_intermediate_checks']

START_DATE = config["general"]["start_date"]
END_DATE = config["general"]["end_date"]

SSH_MODE = config['general']['ssh_mode']

REGION = config['plot_currents']['region']

do_eq = config['general']['do_eq']


#### DIRECTORIES AND INPUT FILE PATTERNS ############
DATA_DIR = os.path.abspath(os.path.join(here, "../../datasets"))
CHECK_DIR = os.path.abspath(os.path.join(here, "../../plots/intermediate_checks"))
FIG_DIR = os.path.abspath(os.path.join(here, "../../plots/maps_currents"))
VALIDATION_DIR = os.path.abspath(os.path.join(here, "../../plots/validation"))
VALIDATION_NCDIR = DATA_DIR + "/OUT/VALIDATION"

if SSH_MODE == "cmems":
    SSH_PATTERN = "ssh_*.nc"
    OUTPUT_DIR = DATA_DIR + "/OUT/CURRENTS/CMEMS"
    SSH_SRC_INTERIM_DIR = DATA_DIR + "/SRC/SSH/CMEMS/INTERIM"
    SSH_SRC_FINAL_DIR = DATA_DIR + "/SRC/SSH/CMEMS/FINAL"
elif SSH_MODE == "neurost":
    SSH_PATTERN = "NeurOST_SSH-SST_*.nc"
    OUTPUT_DIR = DATA_DIR + "/OUT/CURRENTS/NEUROST"
    SSH_SRC_INTERIM_DIR = DATA_DIR + "/SRC/SSH/NEUROST/FINAL"
    SSH_SRC_FINAL_DIR = DATA_DIR + "/SRC/SSH/NEUROST/FINAL"
else:
    SSH_PATTERN = None

SST_SRC_DIR = DATA_DIR + "/SRC/SST/CMC"

WIND_SRC_DIR = DATA_DIR + "/SRC/WIND/ERA5"
WIND_PATTERN = "era5_*.nc"

DRIFTER_SRC_DIR = DATA_DIR + "/SRC/DRIFTERS"

