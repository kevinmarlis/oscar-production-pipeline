# OSCAR

The Ocean Surface Current Analyses Real-time (OSCAR) is a NASA funded research project and global surface current database. The OSCAR ocean surface mixed layer velocities are calculated from satellite-sensed sea surface height gradients, ocean vector winds, and sea surface temperature fields using geostrophy, Ekman, and thermal wind dynamics.

See more information on the algorithm here: https://www.esr.org/data-products/oscar/

Datasets are available on PO.DAAC:  https://podaac.jpl.nasa.gov/cloud-datasets?search=oscar

# Setup Instructions

## Installation
Requires **Python 3.12**. From the repository root, create a virtual environment and install
the package (this pulls in all dependencies, declared in `pyproject.toml`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies installed: xarray, netcdf4, pyyaml, scipy, copernicusmarine, numpy, pandas,
cdsapi, matplotlib, cartopy, scikit-learn, requests, podaac-data-subscriber.

---

## Accounts & Authentication

### 1. NASA Earthdata (PO.DAAC)
- Create an account at [Earthdata Login](https://urs.earthdata.nasa.gov/).
- Create a `.netrc` file in your home directory:

```bash
nano ~/.netrc
```

Add the following (replace with your own credentials):

```
machine urs.earthdata.nasa.gov
  login YOUR_USERNAME
  password YOUR_PASSWORD
```

---

### 2. Copernicus Marine
- Register at [Copernicus Marine](https://marine.copernicus.eu/).
- Add your credentials to the same `.netrc` file:

```
machine auth.marine.copernicus.eu
  login YOUR_USERNAME
  password YOUR_PASSWORD
```

---

### 3. Copernicus Climate Data Store (CDS)
- Create an account at the [CDS Climate Data Store](https://cds.climate.copernicus.eu/).
- Accept the dataset licence(s) under the **Licence** tab.
- Copy your **personal access token** from your [profile page](https://cds.climate.copernicus.eu/how-to-api).
- Create a `.cdsapirc` file:

```bash
nano ~/.cdsapirc
```

Add:

```
url: https://cds.climate.copernicus.eu/api
key: <YOUR-PERSONAL-ACCESS-TOKEN>
```

---

## macOS Note
If you are on macOS and encounter SSL certificate issues, run:

```bash
open "/Applications/Python 3.12/Install Certificates.command"
```

---

## Quick Start
- Change options in `oscar/config/io_config.yaml`
- Option to change paths in `oscar/config/setup.py` (by default, inputs are written to
  `datasets/` and outputs to `plots/` inside the repo; both are git-ignored)
- Once everything is set up (and the venv is activated), run the pipeline via the installed
  console command:

```bash
oscar
```

Equivalently: `python -m oscar.main`
