import numpy as np
import xarray as xr
import glob
import os
import datetime as dt
import re
from ..config.setup import *
from datetime import datetime, timedelta, date
from ..utils.download import *
    
def determine_oscar_mode(dates, ssh_mode):

    oscar_mode = list()
    missing_files = list()

    def exists(root, d8):
        pat = os.path.join(root, "**", f"*{d8}*.nc")
        for p in glob.glob(pat, recursive=True):
            try:
                if os.path.getsize(p) > 0:
                    return True
            except OSError:
                pass
        return False

    # --- Resolve SSH roots by ssh_mode ---
    if ssh_mode == "cmems":
        ssh_downloader  = download_ssh_cmems
    elif ssh_mode == "neurost":
        ssh_downloader  = download_ssh_neurost
    else:
        raise ValueError(f"ssh_mode must be 'cmems' or 'neurost', got: {ssh_mode}")

    # ---------------- Check SSH, SST, wind files, define mode from SSH ----------------
    for d in dates:
        nomissingfile=True
        d8 = d.replace("-", "")[:8] # dates are strings; we also need YYYYMMDD for filename matching 
        
        # Check SST and winds files all here, otherwise try to download
        if not exists(SST_SRC_DIR, d8):
            print(f'Missing SST input file on date: {d}\n')
            missing_files.append(f"sst: {SST_SRC_DIR}/**/*{d8}*.nc")
            nomissingfile=False
        if not exists(WIND_SRC_DIR, d8):
            print(f'Missing Winds input file on date: {d}\n')
            missing_files.append(f"wind: {WIND_SRC_DIR}/**/*{d8}*.nc")
            nomissingfile=False

        # Check SSH files all here and define mode   
        if not nomissingfile: #if there were SST amd/or wind missing, we put oscar_mode to None
            oscar_mode.append('None')   
        elif exists(SSH_SRC_FINAL_DIR, d8):
            oscar_mode.append('final')
        elif exists(SSH_SRC_INTERIM_DIR, d8):
            oscar_mode.append('interim')
        else:
            oscar_mode.append('None')
            missing_files.append(f"ssh: for {d8}")
 
    if missing_files:
        print("\nMissing required files for some dates:\n" + "\n".join(missing_files))           

    return oscar_mode


def get_file_path(date,oscar_mode):
    dt = datetime.strptime(date, "%Y-%m-%d")
    date_str = dt.strftime("%Y%m%d")
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    filename = f"oscar_currents_{oscar_mode}_{date_str}.nc"
    if oscar_mode == "final":
        return os.path.join(OUTPUT_DIR+'/FINAL',str(year),str(month),filename)
    elif oscar_mode == "interim":
        return os.path.join(OUTPUT_DIR+'/INTERIM',str(year),str(month),filename)


def load_ds(date_str, oscar_mode, var):

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    next_day_obj = date_obj + timedelta(days=1)
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    day_str = date_obj.strftime("%Y%m%d")
    
    # Determine source path and file pattern and load a dataset
    if var == "ssh":
        if oscar_mode == "final":
            ssh_src = SSH_SRC_FINAL_DIR
        elif oscar_mode == "interim": 
            ssh_src = SSH_SRC_INTERIM_DIR
   
        search_dir = os.path.join(ssh_src, year, month)
        pattern = SSH_PATTERN.replace("*", f"{day_str}*")
    elif var == "sst":
        search_dir = os.path.join(SST_SRC_DIR, year, month)
        pattern = f"{day_str}*CMC-L4_GHRSST-SSTfnd*.nc"

    elif var == "wind":
        search_dir = os.path.join(WIND_SRC_DIR, year,month)
        pattern = WIND_PATTERN.replace("*", f"{day_str}")

    else:
        raise ValueError(f"Unsupported variable: {var}")
    
    search_path = os.path.join(search_dir, pattern)
    matches = glob.glob(search_path)

    if not matches:
        raise FileNotFoundError(f"No dataset found for {var} on {date_str} at {search_path}")

    if len(matches) > 1:
        raise RuntimeError(f"Multiple files found for {var} on {date_str}. Expected only one:\n{matches}")

    ds = xr.open_dataset(matches[0])


    if var == "ssh":
        ds = ds.rename({'adt': 'ssh'})

    elif var == "sst":
        ds = ds.rename({'analysed_sst': 'sst', 'lat': 'latitude', 'lon': 'longitude'})
        if 'mask' in ds:
            ocean_temp = np.where((ds.mask.values == 1), ds['sst'].values - 273.15, float("NaN"))
        else:
            ocean_temp = ds['sst'].values - 273.15
        ds['sst'].values = ocean_temp

    elif var == "wind":
        ds = ds.rename({'valid_time': 'time'})
        ds = xr.merge([ds['u10'], ds['v10']])

    # === Normalize longitudes ===
    if 'longitude' in ds.coords:
        ds = ds.assign_coords(longitude=(ds.longitude % 360))
        ds = ds.sortby('longitude')

    if var in ["ssh", "sst"]:
        ds = ds[[var]]
    elif var == "wind":
        ds = ds[["u10", "v10"]]
        ds = ds.resample(time='1D').mean()

    if 'time' in ds.dims:
        if ds.sizes['time'] > 1:
            ds = ds.mean(dim='time', keep_attrs=True)

        ds['time'] = [np.datetime64(date_str)] 

    return ds


def write_oscar(refDs,Ug,Uw,Ub,OUTPUTFILE,OUTPUTDIR,SSHLONGDESC,WINDLONGDESC,SSTLONGDESC,OSCARLONGDESC,OSCARSUMMARY,OSCARID,DOI, ssh_mode):
    for ii in range(len(refDs.time)):
        dsii=refDs.sel(time=refDs.time[ii])
        dt = str(dsii.time.to_numpy())
        dateDash = dt[0:10]
        year = dateDash[0:4]
        month = dateDash[5:7]
        day = dateDash[8:10]  
       
        date = year + month + day

        # Year/Month directory structure
        YRMONTHDIR = os.path.join(OUTPUTDIR, year, month)
  

        if not os.path.exists(YRMONTHDIR):
            print("Creating new directory: " + YRMONTHDIR)
            os.makedirs(YRMONTHDIR)

        FILENM = os.path.join(YRMONTHDIR, OUTPUTFILE + date + '.nc')

        if os.path.exists(FILENM):
            print("Removing existing file: " + FILENM)
            os.remove(FILENM)

        write_metadata(dsii,dateDash,Ug[ii,:,:],Uw[ii,:,:],Ub[ii,:,:],FILENM,SSHLONGDESC,WINDLONGDESC,SSTLONGDESC,OSCARLONGDESC,OSCARSUMMARY,OSCARID,DOI, ssh_mode)

        print(f"\n---> Currents file saved at: {FILENM}")
        print("************************************************")
    
        #display('REMOVING md5sum checksum from WRITE_OSCAR')
        # %eval(['! md5sum ',FILENM,' > ',FILENM,'.md5']) 
        
        
def write_metadata(refDs,dateDash,Ug,Uw,Ub,FILENM,SSHLONGDESC,WINDLONGDESC,SSTLONGDESC,OSCARLONGDESC,OSCARSUMMARY,OSCARID,DOI, ssh_mode):

    Ut = Ug+Uw+Ub

    longitude = refDs.coords['longitude'].to_numpy()
    latitude = refDs.coords['latitude'].to_numpy()
    time = np.atleast_1d(refDs.coords['time'].to_numpy())

    u = np.expand_dims(Ut.real.transpose(), 0)
    v = np.expand_dims(Ut.imag.transpose(), 0)
    ug = np.expand_dims(Ug.real.transpose(), 0)
    vg = np.expand_dims(Ug.imag.transpose(), 0)
    uw = np.expand_dims(Uw.real.transpose(), 0)
    vw = np.expand_dims(Uw.imag.transpose(), 0)
    ub = np.expand_dims(Ub.real.transpose(), 0)
    vb = np.expand_dims(Ub.imag.transpose(), 0)

    coords={'time': (['time'], time), 'lon': (['longitude'], longitude), 'lat': (['latitude'], latitude)}

    dsOut = xr.Dataset({'u': (['time', 'longitude', 'latitude'], u), 'v': (['time', 'longitude', 'latitude'], v), \
        'ug' : (['time', 'longitude', 'latitude'], ug), 'vg': (['time', 'longitude', 'latitude'], vg), \
        'uw' : (['time', 'longitude', 'latitude'], uw), 'vw': (['time', 'longitude', 'latitude'], vw), \
        # 'ub' : (['time', 'longitude', 'latitude'], ub), 'vb': (['time', 'longitude', 'latitude'], vb), \
        }, coords=coords)

    addmetadata(dsOut,dateDash,SSHLONGDESC,WINDLONGDESC,SSTLONGDESC,OSCARLONGDESC,OSCARSUMMARY,OSCARID,DOI, ssh_mode)

    epoch = {'units':'days since 1990-01-01'}
    fill = {'_FillValue':-999.0}
    encodings = make_encodings(dsOut)

    dsOut.to_netcdf(FILENM, encoding=encodings)
    
    
def addmetadata(dsOut,dateDash,SSHLONGDESC,WINDLONGDESC,SSTLONGDESC,OSCARLONGDESC,OSCARSUMMARY,OSCARID,DOI, ssh_mode):
    if ssh_mode == 'cmems':
        lat_res = '0.125 degree'
        lon_res = '0.125 degree'
    else:
        lat_res = '0.1 degree'
        lon_res = '0.1 degree'
    # latitude
    dsOut.lat.attrs['long_name'] = 'latitude'
    dsOut.lat.attrs['standard_name'] = 'latitude'
    dsOut.lat.attrs['units'] = 'degrees_north'
    dsOut.lat.attrs['axis'] = 'Y'
    dsOut.lat.attrs['valid_min'] = -89.9375
    dsOut.lat.attrs['valid_max'] =  89.9375
    dsOut.lat.attrs['bounds'] = '[-89.9375,89.9375]'

    # longitude
    dsOut.lon.attrs['long_name'] = 'longitude'
    dsOut.lon.attrs['standard_name'] = 'longitude'
    dsOut.lon.attrs['units'] = 'degrees_east'
    dsOut.lon.attrs['axis'] =  'X'
    dsOut.lon.attrs['valid_min'] = 0.0
    dsOut.lon.attrs['valid_max'] = 359.875
    dsOut.lon.attrs['bounds'] = '[0,359.875]'

    dsOut.time.attrs['long_name'] = 'time centered on the day'
    dsOut.time.attrs['standard_name'] = 'time'
    # these are filled by epoch
    #ncwriteatt(FILENM, 'time', 'units', 'days since 1990-1-1');
    #ncwriteatt(FILENM, 'time', 'calendar', 'julian');
    dsOut.time.attrs['axis'] = 'T'
    dsOut.time.attrs['time_bounds'] = dateDash + ' 00:00:00 to ' + dateDash + ' 23:59:59'
    dsOut.time.attrs['comment'] = 'Data is averaged over the day'

    dsOut.u.attrs['long_name'] = 'zonal total surface current'
    dsOut.u.attrs['standard_name']  = 'eastward_sea_water_velocity'

    dsOut.v.attrs['long_name'] = 'meridional total surface current'
    dsOut.v.attrs['standard_name'] = 'northward_sea_water_velocity'

    for field in ['u', 'v']:
        dsOut[field].attrs['coverage_content_type'] = 'modelResult'
        dsOut[field].attrs['comment'] = 'Velocities are an average over the top 30m of the mixed layer'
        dsOut[field].attrs['source'] = 'SSH source: '+SSHLONGDESC+' ; WIND source: '+WINDLONGDESC+' ; SST source: '+SSTLONGDESC

    dsOut.ug.attrs['long_name'] = 'zonal geostrophic surface current'
    dsOut.ug.attrs['standard_name'] = 'geostrophic_eastward_sea_water_velocity'

    dsOut.vg.attrs['long_name'] = 'meridional geostrophic surface current'
    dsOut.vg.attrs['standard_name'] = 'geostrophic_northward_sea_water_velocity'

    dsOut.uw.attrs['long_name'] = 'zonal Ekman surface current'
    dsOut.uw.attrs['standard_name'] = 'ekman_eastward_sea_water_velocity'

    dsOut.vw.attrs['long_name'] = 'meridional Ekman surface current'
    dsOut.vw.attrs['standard_name'] = 'ekman_northward_sea_water_velocity'

    for field in ['ug', 'vg']:
        dsOut[field].attrs['comment'] = 'Geostrophic velocities calculated from absolute dynamic topography'
        dsOut[field].attrs['source'] = 'SSH source: '+SSHLONGDESC
    
    for field in ['uw', 'vw']:
        dsOut[field].attrs['comment'] = 'Wind driven Ekman velocities'
        dsOut[field].attrs['source'] = 'WIND source: '+WINDLONGDESC
  
    for field in ['u', 'v', 'ug', 'vg', 'uw', 'vw']:
        dsOut[field].attrs['units'] = 'm s-1'
        dsOut[field].attrs['valid_min'] = -3.0
        dsOut[field].attrs['valid_max'] = 3.0
        dsOut[field].attrs['depth'] = '15m'
    
    # GLOBAL ATTRIBUTES
    dsOut.attrs['title'] = OSCARLONGDESC
    dsOut.attrs['summary'] = 'Global, daily, 0.125 degree geostrophic, Ekman, and total mixed layer currents averaged over the top 30m. ' + OSCARSUMMARY;
    dsOut.attrs['keywords'] = 'ocean currents,ocean circulation,surface currents,ekman,geostrophic'
    #dsOut.attrs['keywords_vocabulary'] = 'CF: NetCDF COARDS Climate and Forecast Standard Names'
    dsOut.attrs['Conventions'] = 'CF-1.8 Standard Names v77, ACDD-1.3, netcdf 4.7.3, hdf5 1.8.12'
    dsOut.attrs['id'] = OSCARID
    dsOut.attrs['history'] = 'OSCAR 0.125 degree daily version 3.0 replaces OSCAR 0.25 degree version 2.0'
    dsOut.attrs['source'] = 'OSCAR is based on simplified physics using satellite data; SSH source: '+SSHLONGDESC+' ; WIND source: '+WINDLONGDESC+' ; SST source: '+SSTLONGDESC
    dsOut.attrs['processing_level'] = 'L4'
    #dsOut.attrs['comment'] = 'RECOMMENDED - Provide useful additional information here'
    dsOut.attrs['standard_name_vocabulary'] = 'NetCDF Climate and Forecast (CF) Metadata Convention'
    dsOut.attrs['acknowledgment'] = 'OSCAR products are supported by NASA and may be freely distributed.'
    dsOut.attrs['product_version'] = 'v3.0'
    dsOut.attrs['creator_name'] = 'Kathleen Dohan'
    dsOut.attrs['creator_email'] = 'kdohan@esr.org'
    dsOut.attrs['creator_url'] = 'www.esr.org/research/oscar/'
    dsOut.attrs['creator_type'] = 'person'
    dsOut.attrs['creator_institution'] = 'ESR'
    dsOut.attrs['institution'] = 'Earth & Space Research'
    dsOut.attrs['references'] = 'www.esr.org/research/oscar/, PO.DAAC user guide, DOI: '+DOI
    dsOut.attrs['project'] = 'Ocean Surface Current Analyses Real-time (OSCAR)'
    dsOut.attrs['program'] = 'OSCAR'
    dsOut.attrs['publisher_name'] = 'NASA Physical Oceanography Distributed Active Archive Center (PO.DAAC)'
    dsOut.attrs['publisher_email'] = 'podaac@podaac.jpl.nasa.gov'
    dsOut.attrs['publisher_url'] = 'podaac.jpl.nasa.gov'
    dsOut.attrs['publisher_type'] = 'institution'
    dsOut.attrs['publisher_institution'] = 'PO.DAAC'
    dsOut.attrs['geospatial_lat_min'] = -89.9375
    dsOut.attrs['geospatial_lat_max'] = 89.9375
    dsOut.attrs['geospatial_lat_units'] = "degrees_north"
    dsOut.attrs['geospatial_lat_resolution'] = lat_res
    dsOut.attrs['geospatial_lon_min'] = 0.0
    dsOut.attrs['geospatial_lon_max'] = 359.875
    dsOut.attrs['geospatial_lon_units'] = "degrees_east"
    dsOut.attrs['geospatial_lon_resolution'] = lon_res
    dsOut.attrs['time_coverage_start'] = dateDash + 'T00:00:00'
    dsOut.attrs['time_coverage_end'] = dateDash + 'T23:59:59'
    
    dtNow = dt.datetime.utcnow()
    #Use ISO 8601:2004 for date and time
    dsOut.attrs['date_created'] = dtNow.strftime('%Y-%m-%d')
    

def make_encodings(G, netcdf_fill_value=-999, extra_prints=True):
    # G is the xarray dataset
    
    dv_encoding = {}
    for dv in G.data_vars:
        dv_encoding[dv] = {'zlib':True,
                           'complevel':5,
                           'shuffle':True,
                           'dtype':'float32',
                           '_FillValue':netcdf_fill_value}


        # overwrite default coordinates attribute (PODAAC REQUEST)
        #G[dv].encoding['coordinates'] = dv_coordinate_attrs[dv]

    # PROVIDE SPECIFIC ENCODING DIRECTIVES FOR EACH COORDINATE
    # if extra_prints: print('... creating coordinate encodings')
    coord_encoding = {}

    for coord in G.coords:
        # default encoding: no fill value, float32
        coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}

        if (G[coord].values.dtype == np.int32) or (G[coord].values.dtype == np.int64):
            coord_encoding[coord]['dtype'] ='int32'

        if coord == 'time' or coord == 'time_bnds':
            coord_encoding[coord]['dtype'] ='int32'

            if 'units' in G[coord].attrs:
                # apply units as encoding for time
                coord_encoding[coord]['units'] = G[coord].attrs['units']
                # delete from the attributes list
                del G[coord].attrs['units']

        elif coord == 'time_step':
            coord_encoding[coord]['dtype'] ='int32'

    # MERGE ENCODINGS for coordinates and variables
    encoding = {**dv_encoding, **coord_encoding}

    return encoding
    
    
def extract_date(path):
    filename = os.path.basename(path)
    date_str = ''.join(filter(str.isdigit, filename))
    year = date_str[0:4]
    month = date_str[4:6]
    return year, month
    
    
def get_existing_dates(OUTPUT_DIR, start_date, end_date):
    existing = set()
    start = np.datetime64(start_date)
    end = np.datetime64(end_date)

    current = start
    while current <= end:
        year = str(current.astype('datetime64[Y]'))[:4]
        month = f"{int(current.astype('datetime64[M]').astype(int) % 12 + 1):02}"

        year_month_path = os.path.join(OUTPUT_DIR, year, month)

        if os.path.exists(year_month_path):
            for root, _, files in os.walk(year_month_path):
                for f in files:
                    match = re.search(r"(\d{8})", f)
                    if match:
                        try:
                            dt = np.datetime64(f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}")
                            existing.add(dt)
                        except Exception:
                            pass
        current = (current.astype('datetime64[M]') + 1).astype('datetime64[D]')
    return existing
    
    
def get_dates_to_process(start_date, end_date, OUTPUT_DIR, override=False):
    all_dates = np.arange(np.datetime64(start_date), np.datetime64(end_date) + np.timedelta64(1, 'D'))
    existing_dates = get_existing_dates(OUTPUT_DIR+'/FINAL', start_date, end_date) #we check existing final dates

    a=[str(d) for d in all_dates]
    
    if override: #we recompute all currents files if overwrite=true
        p=[str(d) for d in all_dates]
        if not p:
            print('NO DATES FOUND')
        else:
            print(p)
        return a, p

    else: #we recompute only the currents files that are not already computed in final (ie interim currents and/or non existing)
        p=[str(d) for d in all_dates if d not in existing_dates]
        if not p:
            print('NO DATES FOUND')
        else:
            print(p)
        return a, p
    
    
