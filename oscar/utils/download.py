import os
import glob
import subprocess
from datetime import datetime, timedelta
import requests
from copernicusmarine import subset
import cdsapi
from ..config.setup import *
from .date_helpers import get_month_info

FINAL_ID   = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
INTERIM_ID = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"
MIN_BYTES    = 10_000                   # guard against tiny placeholder files


# --- Helpers ----------------------------------------------------------

def _coerce_dt(d):
    """Return a datetime.datetime from either a datetime/date or 'YYYY-MM-DD' string."""
    if hasattr(d, "strftime"):
        return d
    return datetime.strptime(d, "%Y-%m-%d")


def _has_nc_for_date(folder: str, dt) -> bool:
    """
    True if there is at least one non-empty .nc file in `folder`
    whose name contains YYYYMMDD for `dt` (works for NEUROST/CMC patterns).
    """
    d8 = _coerce_dt(dt).strftime("%Y%m%d")
    pattern = os.path.join(folder, f"*{d8}*.nc")
    for p in glob.glob(pattern):
        try:
            if os.path.getsize(p) > 0:
                return True
        except OSError:
            pass
    return False


def _nonempty(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) >= MIN_BYTES
    except OSError:
        return False
        
        
# --- CMEMS SSH --------------------------------------------------------

def download_ssh_cmems(dates):
    
    print('\nDOWNLOADING CMEMS SSH...\n')
    for d in dates:
        dt = _coerce_dt(d)
        y  = dt.strftime("%Y")
        m  = dt.strftime("%m")
        d8 = dt.strftime("%Y%m%d")

        final_dir   = os.path.join(SSH_SRC_FINAL_DIR,   y, m)
        interim_dir = os.path.join(SSH_SRC_INTERIM_DIR, y, m)
        final_file   = os.path.join(final_dir,   f"ssh_{d8}.nc")
        interim_file = os.path.join(interim_dir, f"ssh_{d8}.nc")

        os.makedirs(final_dir, exist_ok=True)
        if _nonempty(final_file): #if final file already exists 
            if not OVERWRITE_DOWNLOAD: #and we don't want to overwrite, we skip
                print(f"CMEMS skip (final already exists):   {final_file}")
                continue

        # ---- Try FINAL first otherwise: if 1. final file exists and we want to overwrite; 2. if final file doesn't exist
        tried_final = False
        try:
            try: os.remove(final_file) #if final file already exists but we want to overwrite we delete it first
            except OSError: pass
            subset(
                dataset_id=FINAL_ID,
                variables=[
                    "adt","err_sla","err_ugosa","err_vgosa","flag_ice","sla",
                    "tpa_correction","ugos","ugosa","vgos","vgosa"
                ],
                minimum_longitude=-179.9375,
                maximum_longitude= 179.9375,
                minimum_latitude= -89.9375,
                maximum_latitude=  89.9375,
                start_datetime=f"{d8}T00:00:00",
                end_datetime=f"{d8}T00:00:00",
                output_directory=final_dir,
                output_filename=f"ssh_{d8}.nc",
                file_format="netcdf",
            )
            if _nonempty(final_file):
                print(f"CMEMS FINAL saved: {final_file}\n")
                continue
            else:
                # clean up tiny/empty, fall through to interim
                try: os.remove(final_file)
                except OSError: pass
                print(f"FINAL returned empty/invalid file for {d8}, trying INTERIM…")
        except Exception as e:
            print(f"FINAL not available for {d8}: {e}. Trying INTERIM…")


        if _nonempty(interim_file):
            if not OVERWRITE_DOWNLOAD: #if interim file already exists and we don't want to overwrite, we skip
                print(f"CMEMS skip (interim already exists): {interim_file}")
                continue
                
        # ---- Fall back to INTERIM if 1. final and interim files don't exist; 2. if final file doesn't exist but interim yes but we want to overwrite
        os.makedirs(interim_dir, exist_ok=True)

        try:
            try: os.remove(interim_file) #if interim file already exists but we want to overwrite we delete it first
            except OSError: pass
            subset(
                dataset_id=INTERIM_ID,
                variables=[
                    "adt","err_sla","err_ugosa","err_vgosa","flag_ice","sla","ugos","ugosa","vgos","vgosa"
                ],
                minimum_longitude=-179.9375,
                maximum_longitude= 179.9375,
                minimum_latitude= -89.9375,
                maximum_latitude=  89.9375,
                start_datetime=f"{d8}T00:00:00",
                end_datetime=f"{d8}T00:00:00",
                output_directory=interim_dir,
                output_filename=f"ssh_{d8}.nc",
                file_format="netcdf",
            )
            if _nonempty(interim_file):
                print(f"CMEMS INTERIM saved: {interim_file}\n")
                continue
            else:
                try: os.remove(interim_file)
                except OSError: pass
                raise RuntimeError(f"INTERIM returned empty/invalid file for {d8}.")
        except Exception as e:
            print(f"FINAL not available for {d8}: {e}")
        

        



# --- NEUROST SSH (PO.DAAC CLI) ---------------------------------------

def download_ssh_neurost(dates):
    """
    Download NEUROST SSH/SST L4 files for each date via podaac-data-downloader CLI.
    Respects skip_existing_neurost_download and keeps any provider filenames.
    """
    os.makedirs(SSH_SRC_FINAL_DIR, exist_ok=True)
    
    print('\n\nDOWNLOADING NEUROST SSH...\n')
    for date in dates:
        start = _coerce_dt(date)
        end   = start + timedelta(days=1)
        start_str = start.strftime("%Y-%m-%dT00:00:00Z")
        end_str   = end.strftime("%Y-%m-%dT00:00:00Z")

        year  = start.strftime("%Y")
        month = start.strftime("%m")
        target_folder = os.path.join(SSH_SRC_FINAL_DIR, year, month)

        # skip if any .nc for this YYYYMMDD is already present and we don't want to overwrite
        if _has_nc_for_date(target_folder, start): #if file already exists
            if not OVERWRITE_DOWNLOAD: #and we don't want to overwrite, we skip
                print(f"NEUROST skip (file already existing for {start.strftime('%Y-%m-%d')}: {target_folder})")
                continue

        
        os.makedirs(target_folder, exist_ok=True)
        d8 = _coerce_dt(start).strftime("%Y%m%d")
        pattern = os.path.join(target_folder, 'NeurOST_SSH-SST'+f"*{d8}*.nc")
        try: 
            for p in glob.glob(pattern): #if file already exists but we want to overwrite we delete it first
                os.remove(p)
        except OSError: pass
        try:
            cmd = [
                "podaac-data-downloader",
                "-c", "NEUROST_SSH-SST_L4_V2024.0",
                "-d", target_folder,
                "--start-date", end_str, #issue when downloading, need to add a day to the date
                "--end-date",   end_str,
                "-e", ".nc",
            ]
            subprocess.run(cmd, check=True)
            if _has_nc_for_date(target_folder, start): #if file exists
                print(f"NEUROST saved in: {target_folder}")
            else:
                print(f"Failed to download NEUROST for {d8}")
            print('')
        except Exception as e:
            print(f"Failed to download NEUROST for {d8}: {e}")

        # clean up extra .txt (manifest/citation) files, if any
        for f in os.listdir(target_folder):
            if f.endswith(".txt"):
                try:
                    os.remove(os.path.join(target_folder, f))
                except OSError:
                    pass


# --- ERA5 WIND (CDS API) ---------------------------------------------

def download_wind_era5(dates):
    """
    Download daily ERA5 10m U/V wind files as single NetCDF per day.
    Respects skip_existing_era5_download and writes 'era5_YYYYMMDD.nc'.
    """
    client = cdsapi.Client()
    
    print('\n\nDOWNLOADING ERA5 Winds...\n')
    for date in dates:
        dt = _coerce_dt(date)
        year  = dt.strftime("%Y")
        month = dt.strftime("%m")
        day   = dt.strftime("%d")

        out_dir  = os.path.join(WIND_SRC_DIR, year, month)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"era5_{year}{month}{day}.nc")

        if os.path.exists(out_file) and os.path.getsize(out_file) > 0: #if file already exists
            if not OVERWRITE_DOWNLOAD: #and we don't want to overwrite, we skip 
                print(f"ERA5 skip (file already existing: {out_file})")
                continue
                
        try: os.remove(out_file)  #if file already exists but we want to overwrite we delete it first
        except OSError: pass
        
        try:
            request = {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": year,
                "month": month,
                "day": day,
                "time": [f"{hour:02d}:00" for hour in range(24)],
                "format": "netcdf",
            }
            client.retrieve("reanalysis-era5-single-levels", request, out_file)
            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                print(f"ERA5 saved: {out_file}")
            else:
                print(f"Failed to download ERA5: {out_file}")
        except Exception as e:
            print(f"Failed to download ERA5: {out_file}: {e}")
        print('')


# --- CMC SST (PO.DAAC CLI) -------------------------------------------

def download_sst_cmc(dates):
    os.makedirs(SST_SRC_DIR, exist_ok=True)
    
    print('\n\nDOWNLOADING CMC SST...\n')
    for date in dates:
        dt = _coerce_dt(date)

        # new version first choice based on date, with the other as old version fallback
        collection   = "CMC0.2deg-CMC-L4-GLOB-v2.0" if dt < datetime(2016, 1, 1) else "CMC0.1deg-CMC-L4-GLOB-v3.0"

        start = dt
        end   = dt + timedelta(days=1)
        start_str = start.strftime("%Y-%m-%dT00:00:00Z")
        end_str   = start.strftime("%Y-%m-%dT23:59:59Z")

        year  = dt.strftime("%Y")
        month = dt.strftime("%m")
        target_folder = os.path.join(SST_SRC_DIR, year, month)
        print(target_folder)

        # skip if any .nc for this YYYYMMDD is already present and we don't want to overwrite
        if _has_nc_for_date(target_folder, start): #if file already exists
            if not OVERWRITE_DOWNLOAD: #and we don't want to overwrite, we skip
                print(f"CMC SST skip (file already existing for {dt.strftime('%Y-%m-%d')}: {target_folder})")
                continue

        os.makedirs(target_folder, exist_ok=True)
        d8 = _coerce_dt(start).strftime("%Y%m%d")
        pattern = os.path.join(target_folder, f"*{d8}*.nc")
        try: 
            for p in glob.glob(pattern): #if file already exists but we want to overwrite we delete it first
                os.remove(p)
        except OSError: pass
        
        # --- Download ---
        try:
            cmd = [
                "podaac-data-downloader",
                "-c", collection,
                "-d", target_folder,
                "--start-date", end_str, #only way to only download one file
                "--end-date",   end_str,
                "-e", ".nc",
            ]
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"CMC SST failed to download for {dt.strftime('%Y-%m-%d')} ({collection}): {e}")


        # Final check
        if not _has_nc_for_date(target_folder, start):
            print(
                f"CMC SST failed to download for {dt.strftime('%Y-%m-%d')}\n"
            )
        else:
            print(f"CMC SST saved in: {target_folder}\n")
            print('')

        # clean up extra .txt (manifest/citation) files, if any
        for f in os.listdir(target_folder):
            if f.endswith(".txt"):
                try:
                    os.remove(os.path.join(target_folder, f))
                except OSError:
                    pass
                    

# --- DRIFTER DATA -------------------------------------------
def get_drifter_data(year_month):
    print('\n\nDOWNLOADING DRIFTER DATA...\n')

    year, month, start_day, end_day = get_month_info(year_month)
    # Construct date range
    start_str = f"{year}-{month}-{start_day}T00:00:00Z"
    end_str = f"{year}-{month}-{end_day}T23:59:59Z"

    # Save location
    target_dir = os.path.join(DRIFTER_SRC_DIR)

    filename = f"drifter_6hour_qc_{year}{month}.nc"
    target_path = os.path.join(target_dir, filename)

    # Skip if file exists and we don't want to overwrite or we don't do the validation
    if os.path.exists(target_path) and not OVERWRITE_DOWNLOAD:
        print(f"Drifter data skip (file already existing: {target_path})")
        return target_path
    if not DO_VALIDATION:
        print(f"Drifter data not needed - no validation")
        return target_path

    os.makedirs(target_dir, exist_ok=True)

    # ERDDAP subset URL for that year/month
    base_url = "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc.nc"
    query = (
        "?ID,time,latitude,longitude,lon360,ve,vn,sst,err_lat,err_lon,err_sst,"
        "typedeath,deploy_date,deploy_lat,deploy_lon,start_date,start_lat,start_lon,"
        "end_date,end_lat,end_lon,drogue_lost_date"
        f"&time>={start_str}&time<={end_str}"
    )
    full_url = base_url + query

    try:
        response = requests.get(full_url)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(response.content)
        print(f"Drifter data saved in: {target_path}\n")
    except Exception as e:
        print(f"Drifter data failed to download for {year} {month}: {e}")
        return None

    return target_path


def download_drifter_data(dates):
    """
    Given a list of daily dates (YYYY-MM-DD), download drifter data 
    grouped by unique months.
    """
    if not isinstance(dates, (list, tuple)):
        raise ValueError("dates must be a list or tuple of 'YYYY-MM-DD' strings.")

    # Normalize and extract unique months
    months = set()
    for s in dates:
        dt = datetime.strptime(s.strip(), "%Y-%m-%d")
        months.add(dt.strftime("%Y-%m"))

    # Download each unique month
    results = []
    for ym in sorted(months):
        results.append(get_drifter_data(ym))

    return results