import numpy as np
from .computation.interp_and_grad import *
from .computation.compute_currents import *
from .utils.data_utils import *
from .config.setup import *  # contains: compute_currents, plot_currents, validate_currents, start_date, end_date, etc.
from .analysis.plot_currents import *
from .analysis.drifter_func import *
from .utils.download import *
import warnings
from .analysis.prepare_pdf import *
import os, glob
from datetime import datetime
import sys

warnings.filterwarnings("ignore")

def run_compute_currents(dates_to_process, oscar_mode):

    missingfilesdates=list()
    for d in range(0,len(dates_to_process)):
        date = dates_to_process[d]
        datedatetime = datetime.strptime(date, "%Y-%m-%d")
        
        if oscar_mode[d]=='None':
            print(f"\n---> At least one input file is not available on {date}, we skip that day")
            missingfilesdates.append(date)
            continue
        else:
            year  = datetime.strptime(date, "%Y-%m-%d").strftime("%Y")
            month = datetime.strptime(date, "%Y-%m-%d").strftime("%m")
            day   = datetime.strptime(date, "%Y-%m-%d").strftime("%d") 
            filename = os.path.join(OUTPUT_DIR+'/INTERIM', f'{year}/{month}/oscar_currents_interim_{year}{month}{day}.nc')
            
            if oscar_mode[d]=='interim' and os.path.isfile(filename) and not OVERWRITE_CURRENT: #in case the mode is interim and interim file already exists, unless we overwrite (get_dates_to_process only checked for existing final files as we still want to see if all interim currents have updated final input files)
                print(f'The interim file for {dates_to_process[d]} already exists, no final input file available, no overwrite, skip')
                
            else:
                
                print(f"\nInterpolations and gradients computation for {dates_to_process[d]}...")
                # SSH
                ssh_ds = load_ds(dates_to_process[d], oscar_mode[d], var='ssh')
                ssh_ds = interpolate_dataset(ssh_ds)
                if DO_CHECKS:
                    plot_interp(ssh_ds, f'interpolated SSH - {year}{month}{day} (m)', f'interp_ssh_{year}{month}{day}.png', os.path.join(CHECK_DIR,SSH_MODE.upper(),'interpolations'),'ssh',-2,2)
                ssh_ds = calculate_gradient(ssh_ds, 'ssh')
                if DO_CHECKS:
                    plot_grad(ssh_ds, f'SSH gradients x and y - {year}{month}{day}', f'grad_ssh_{year}{month}{day}.png', os.path.join(CHECK_DIR,SSH_MODE.upper(),'gradients'),'sshx','sshy',-0.000005,0.000005)

                # SST
                sst_ds = load_ds(dates_to_process[d], oscar_mode[d], var='sst')
                sst_ds = interpolate_dataset(sst_ds)
                if DO_CHECKS:
                    plot_interp(sst_ds, f'interpolated SST - {year}{month}{day} (degC)', f'interp_sst_{year}{month}{day}.png', os.path.join(CHECK_DIR,SSH_MODE.upper(),'interpolations'),'sst',-2,30)
                sst_ds = calculate_gradient(sst_ds, 'sst')
                if DO_CHECKS:
                    plot_grad(sst_ds, f'SST gradients x and y - {year}{month}{day}', f'grad_sst_{year}{month}{day}.png', os.path.join(CHECK_DIR,SSH_MODE.upper(),'gradients'),'sstx','ssty',-0.00005,0.00005)

                # Wind
                wind_ds = load_ds(dates_to_process[d], oscar_mode[d], var='wind')
                wind_ds = interpolate_dataset(wind_ds)
                if DO_CHECKS:
                    plot_interp(wind_ds, f'interpolated u winds - {year}{month}{day} (m/s)', f'interp_uwind_{year}{month}{day}.png', os.path.join(CHECK_DIR,SSH_MODE.upper(),'interpolations'),'u10',-15,15)
                    plot_interp(wind_ds, f'interpolated v winds - {year}{month}{day} (m/s)', f'interp_vwind_{year}{month}{day}.png', os.path.join(CHECK_DIR,SSH_MODE.upper(),'interpolations'),'v10',-15,15)

                # Compute and write OSCAR
                ref_ds = ssh_ds.drop_vars(['sshx', 'sshy'])
                print(f"Currents computation for {dates_to_process[d]}...")
                Ug, Uw, Ub = compute_surface_currents(ssh_ds, wind_ds, sst_ds, do_eq)
                print(f"\n---> Finished computing currents for {dates_to_process[d]}")

                save_file = f"oscar_currents_{oscar_mode[d]}_"

                wind_long_desc = "ECMWF ERA5 10m wind DOI: 10.24381/cds.adbb2d47"
                if datedatetime < datetime(2016, 1, 1):
                    sst_long_desc =  "CMC 0.2 deg SST V2.0 DOI: 10.5067/GHCMC-4FM02"
                else:
                    sst_long_desc =  "CMC 0.1 deg SST V3.0 DOI: 10.5067/GHCMC-4FM03"
                if SSH_MODE == 'cmems':
                    if oscar_mode[d]=='final':
                        outputdir = OUTPUT_DIR+f'/{oscar_mode[d].upper()}'
                        ssh_long_desc = "CMEMS SSALTO/DUACS SEALEVEL_GLO_PHY_L4_MY_008_047 DOI: 10.48670/moi-00148"
                        oscar_long_desc = f"Ocean Surface Current Analyses Real-time (OSCAR) Surface Currents - {oscar_mode[d].upper()} 0.125 Degree (Version 3.0)"
                        oscar_summary = "Highest quality OSCAR product."
                        oscar_id = f"OSCAR_L4_OC_{oscar_mode[d].upper()}_V3.0"
                        doi =  "10.5067/OSCAR-25F20"
                    elif oscar_mode[d]=='interim':
                        outputdir = OUTPUT_DIR+f'/{oscar_mode[d].upper()}'
                        ssh_long_desc = "CMEMS SSALTO/DUACS SEALEVEL_GLO_PHY_L4_NRT_OBSERVATIONS_008_046 DOI: 10.48670/moi-00149"
                        oscar_long_desc = f"Ocean Surface Current Analyses Real-time (OSCAR) Surface Currents - {oscar_mode[d].upper()} 0.125 Degree (Version 3.0)"
                        oscar_summary = "Lower quality than final currents."
                        oscar_id = f"OSCAR_L4_OC_{oscar_mode[d].upper()}_V3.0"
                        doi =  "10.5067/OSCAR-25I20"
                elif SSH_MODE == 'neurost':
                    outputdir = OUTPUT_DIR+f'/{oscar_mode[d].upper()}'
                    ssh_long_desc = "Daily NeurOST L4 Sea Surface Height (NEUROST_SSH-SST_L4_V2024.0) DOI: 10.5067/NEURO-STV24"
                    oscar_long_desc = "Ocean Surface Current Analyses Real-time (OSCAR) Surface Currents - 0.01 Degree (Version 3.0)"
                    oscar_summary = ""
                    oscar_id = ""
                    doi =  ""
                else:
                    print(f'Unknown ssh_mode, should be cmems or neurost - EXIT')
                    sys.exit()
                write_oscar(ref_ds, Ug, Uw, Ub, save_file, outputdir,
                                      ssh_long_desc, wind_long_desc, sst_long_desc,
                                      oscar_long_desc, oscar_summary, oscar_id,
                                      doi, SSH_MODE)
                                  
    print(f"\nALL CURRENTS COMPUTATION FOR {dates_to_process[0]} - {dates_to_process[-1]} IS COMPLETED")
    if DO_CHECKS:
        print(f'Interpolation and gradient plots saved in {os.path.join(CHECK_DIR,SSH_MODE.upper())}')
    if missingfilesdates:
        print(f'EXCEPT FOR DATES {missingfilesdates}\n')


def run_plotting(dates,oscar_mode):

    for d in range(0,len(dates)):
        ds_path= get_file_path(dates[d],oscar_mode[d])
        if not ds_path:
            print(f'Current file not found for: {dates[d]}')
            continue
        save_dir = os.path.join(FIG_DIR, SSH_MODE.upper(), REGION)
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(FIG_DIR, SSH_MODE.upper(), REGION, os.path.join(os.path.basename(ds_path).replace(".nc", ".png")))

        if OVERWRITE_CURRENT or (not OVERWRITE_CURRENT and not os.path.exists(filename)):
            try: 
                os.remove(os.path.join(filename.replace("final", "interim"))) #if interim figure exists, we delete before creating final
                print(f'Deleting {os.path.join(filename.replace("final", "interim"))}')
            except OSError: pass
            title = f"OSCAR Surface Currents - {REGION.upper()} - {dates[d]}"
            ds = xr.open_dataset(ds_path)
            u, v, lon, lat, speed, lon2d, lat2d = extract_data(ds)
            plot_currents(ds_path, filename, lon2d, lat2d, u, v, speed, title, SSH_MODE)
        else:
            print(f'Figure already exists and no data/currents updates - skip {filename}')
    print(f'Currents maps saved in {os.path.join(FIG_DIR, SSH_MODE.upper(), REGION)}')


def run_validation(dates,oscar_mode):
    
    #output validation pdf file
    years, months, days = dates[0].split("-")
    yeare, monthe, daye = dates[-1].split("-")
    temporal_range = f'{years}{months}{days}-{yeare}{monthe}{daye}'
    today = datetime.today().strftime("%Y-%m-%d").replace("-", "")
    validation_dir = os.path.join(VALIDATION_DIR)
    os.makedirs(validation_dir, exist_ok=True)
    validation_pdf_path = os.path.join(VALIDATION_DIR, f'Validation_report_{SSH_MODE.upper()}_{years}{months}{days}-{yeare}{monthe}{daye}_{today}.pdf')
    
    #dates start and end
    years, months, days = dates[0].split("-")
    yeare, monthe, daye = dates[-1].split("-")
    
    #list all the oscar and drifter files
    drifterfilelist=list()
    for d in range(0,len(dates)): 
        date=dates[d]
        year, month, day = date.split("-")
        oscar_file = os.path.join(OUTPUT_DIR,oscar_mode[d],year,month,f'oscar_currents_{oscar_mode[d]}_{year}{month}{day}.nc')
        drifter_file = os.path.join(DRIFTER_SRC_DIR,f'drifter_6hour_qc_{year}{month}.nc')
        if not os.path.isfile(oscar_file):
            print(f'\nWARNING!! Missing OSCAR files to perform the validation - EXIT')
            sys.exit()
        if os.path.isfile(drifter_file):
            if drifter_file not in drifterfilelist:
                drifterfilelist.append(drifter_file)
        else:
            print(f'\nWARNING!! Missing DRIFTER file for: {year}-{month} to perform the validation - EXIT')
            sys.exit()


    #make daily average for each drifter id, if not done already
    for drifter in drifterfilelist: 
        year = drifter[-9:-5]
        month = drifter[-5:-3]
        daily_avg_by_id_dir = os.path.join(DRIFTER_SRC_DIR,"daily_avg_by_id",year,month)
        daily_avg_dir = os.path.join(DRIFTER_SRC_DIR,"daily_avg",year,month)
        os.makedirs(daily_avg_by_id_dir, exist_ok=True)
        os.makedirs(daily_avg_dir, exist_ok=True)

        if not os.path.exists(os.path.join(daily_avg_by_id_dir,'.daily_avg_by_id_done')): #daily averages per drifter id not redone if it was previously done, can add an overwrite function here
            with xr.open_dataset(drifter) as ds_month:
                #creates nc file with daily average for each drifter id in daily_avg_by_id
                get_unique_drifter_daily_avg(ds_month, daily_avg_by_id_dir)
            open(os.path.join(daily_avg_by_id_dir,'.daily_avg_by_id_done'), "w").close()
        else:
            print(f'Daily averages DRIFTER by ID already done for {year}-{month} - skip')
     
    drifter_vs_oscar_dslist=list()
    for d in range(0,len(dates)): 
        date=dates[d]
        year, month, day = date.split("-")
        daily_avg_by_id_dir = os.path.join(DRIFTER_SRC_DIR,"daily_avg_by_id",year,month)
        daily_avg_dir = os.path.join(DRIFTER_SRC_DIR,"daily_avg",year,month)

        #create one file per day with all drifters
        dailyfilename = os.path.join(daily_avg_dir, f"drifter_{year}{month}{day}.nc")
        if not os.path.isfile(dailyfilename):
            #creates drifter daily_avg files in daily_avg folder which has for each day, all the drifters id, their lat, lon, ve, vn etc.
            get_daily_avg_all_drifters(daily_avg_by_id_dir, daily_avg_dir, date, dailyfilename)
        else:
            print(f'Daily average DRIFTER file already existing for {year}-{month}-{day} - skip')
        
        #interpolate OSCAR on daily drifter    
        ds_drifter = xr.open_dataset(dailyfilename)
    
        ds = xr.open_dataset(os.path.join(OUTPUT_DIR,oscar_mode[d],year,month,f'oscar_currents_{oscar_mode[d]}_{year}{month}{day}.nc'))
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
        ds = ds.set_coords(['lat', 'lon'])
        u = ds['u'].transpose('time', 'lat', 'lon')
        u = u.assign_coords(lat=ds['lat'], lon=ds['lon'])
    
        drifter_vs_oscar_ds = interpolate_oscar_to_drifters(ds, ds_drifter)
        drifter_vs_oscar_dslist.append(drifter_vs_oscar_ds) # otherwise we keep appending

    drifter_vs_oscar_dslist = xr.concat(drifter_vs_oscar_dslist, dim='drifter')
        
    if SAVE_COLOCATION:
        #output colocation nc file    
        validation_ncdir = os.path.join(VALIDATION_NCDIR, SSH_MODE.upper()) 
        os.makedirs(validation_ncdir, exist_ok=True)
        validation_ncpath = os.path.join(validation_ncdir,f'drifters_oscar_colocation_{years}{months}{days}_{yeare}{monthe}{daye}_{today}.nc') # for the last month
        try:
            pattern = os.path.join(validation_ncdir,f'drifters_oscar_colocation_{years}{months}{days}_{yeare}{monthe}{daye}_*.nc')
            for file in glob.glob(pattern):
                os.remove(file) # we remove any previous nc file
        except OSError: pass
        drifter_vs_oscar_dslist.to_netcdf(validation_ncpath, mode = 'w')
        print(f'OSCAR/DRIFTER colocation ncfile saved at: {validation_ncpath}')
        
    corr_map = compute_binned_correlations(
        drifter_vs_oscar_dslist,
        components=('u','v'),
        lat_range=(-90, 90),
        lon_range=(-180, 180),
        bin_size=2)    
    
    captions = [
        "Scatter plots of (left) U OSCAR vs U Drifters and (right) V OSCAR vs V Drifter",
        "Maps of 2x2 degree binned correlations between U OSCAR and U Drifter",
        "Maps of 2x2 degree binned correlations between V OSCAR and V Drifter",
        "Density histogram of U OSCAR vs U Drifter",
        "Density histogram of V OSCAR vs V Drifter",
        "Linear regression slope of U OSCAR vs U Drifter",
        "Linear regression slope of V OSCAR vs V Drifter",
        "Map of RMSD between U OSCAR and U Drifter",
        "Map of RMSD between V OSCAR and V Drifter",
        "Map of Pearson correlation coefficient between U OSCAR and U Drifter",
        "Map of Pearson correlation coefficient between V OSCAR and V Drifter"
    ]
    
    figs = []
    figs.append(plot_velocity_comparison_scatter(drifter_vs_oscar_dslist, temporal_range))
    figs.append(plot_binned_correlation_map(temporal_range, corr_map['corr_u'], 'u', zoom_extent=None))
    figs.append(plot_binned_correlation_map(temporal_range, corr_map['corr_v'], 'v', zoom_extent=None))
    figs.append(plot_density(drifter_vs_oscar_dslist, temporal_range, 'u'))
    figs.append(plot_density(drifter_vs_oscar_dslist, temporal_range, 'v'))
    figs.append(make_slope_map(drifter_vs_oscar_dslist, 'u', temporal_range))
    figs.append(make_slope_map(drifter_vs_oscar_dslist, 'v', temporal_range))
    figs.append(make_rms_map(drifter_vs_oscar_dslist, 'u', temporal_range))
    figs.append(make_rms_map(drifter_vs_oscar_dslist, 'v', temporal_range))
    figs.append(make_residual_corr_map(drifter_vs_oscar_dslist, 'u', temporal_range))
    figs.append(make_residual_corr_map(drifter_vs_oscar_dslist, 'v', temporal_range))
    
    try: #remove existing pdf file for this time range
        pattern =os.path.join(VALIDATION_DIR, f'Validation_report_{SSH_MODE.upper()}_{years}{months}{days}-{yeare}{monthe}{daye}_*.pdf')
        for file in glob.glob(pattern):
            os.remove(file) # we remove any previous nc file
    except OSError: pass
    save_plots_to_pdf(figs, captions,
                    pdf_path=validation_pdf_path,
                    metadata={"Title":f"Comparison OSCAR (from {SSH_MODE}) and Drifters - {temporal_range}","Date":datetime.today().strftime("%Y-%m-%d")})

    print(f'Validation pdf file saved at: {validation_pdf_path}')
    
    return


def run_download(dates):

    if SSH_MODE == "cmems":
        download_ssh_cmems(dates)
    elif SSH_MODE  == "neurost":
        download_ssh_neurost(dates)
    download_wind_era5(dates)
    download_sst_cmc(dates)
    download_drifter_data(dates)

    return


def main():
    np.seterr(divide='ignore', invalid='ignore', over='ignore')

    print(f'\n\n************************************************')
    print(f'RUNNING THE OSCAR ALGORITHM from {START_DATE} to {END_DATE}')
    print(f'...SSH input from {SSH_MODE.upper()}') 
    print(f'...overwrite download {OVERWRITE_DOWNLOAD}') 
    print(f'...overwrite current computation {OVERWRITE_CURRENT}') 
    print(f'...plotting intermediate checks {DO_CHECKS}') 
    print(f'...plotting daily current maps {PLOT_CURRENTS} in {REGION}') 
    print(f'...validation {DO_VALIDATION}') 
    print(f'...save colocation files {SAVE_COLOCATION}') 
            
    if OVERWRITE_DOWNLOAD and not OVERWRITE_CURRENT:
        print("\nWARNING!! overwrite_currents cannot be False while overwrite_download is True - EXIT")
        sys.exit()
            
    if  (OVERWRITE_CURRENT or OVERWRITE_DOWNLOAD or DO_VALIDATION) and (PLOT_CURRENTS and REGION!='global'):
        print("\nWARNING!! The region option is only for the currents plotting. The input data download, currents computation, and validation will be done globally")
            
    if not PLOT_CURRENTS and REGION!='global':
        print("\nWARNING!! The region option is only for the currents plotting. For currents computation and validation, use 'global' - EXIT")
        sys.exit()
            
    if SAVE_COLOCATION and not DO_VALIDATION:
        print("\nWARNING!! Colocation files will not be made and saved if do_validation is False")
        sys.exit()
    
    print("\n\n************************************************")    
    start_dt = np.datetime64(START_DATE)
    end_dt = np.datetime64(END_DATE)
    print('\n\nCOMPUTE CURRENTS ONLY FOR...')
    all_dates, dates_to_process = get_dates_to_process(start_dt, end_dt, OUTPUT_DIR, OVERWRITE_CURRENT) #depends on the output files already created

    if len(dates_to_process) != 0:
        print("\n\n************************************************")
        print("STARTING DOWNLOADING...\n")
        run_download(dates_to_process)
        
        print("\n\n************************************************")
        print("STARTING COMPUTING CURRENTS...\n")
        np.seterr(divide='ignore', invalid='ignore', over='ignore')
        start_dt = np.datetime64(START_DATE)
        end_dt = np.datetime64(END_DATE)
        oscar_mode = determine_oscar_mode(dates_to_process, SSH_MODE)
        run_compute_currents(dates_to_process, oscar_mode)


    if PLOT_CURRENTS: #Done on dates from start to end unless file already exists and overwrite_current is False
        if len(all_dates) != 0:
            print("\n\n************************************************")
            print("STARTING PLOTTING CURRENTS...\n")
            oscar_mode = determine_oscar_mode(all_dates, SSH_MODE)
            run_plotting(all_dates,oscar_mode)

    if DO_VALIDATION: #Done on dates from start to end unless file already exists and overwrite_current is False
        if len(all_dates) != 0:
            print("\n\n************************************************")
            print("STARTING VALIDATION...\n")
            oscar_mode = determine_oscar_mode(all_dates, SSH_MODE)
            run_validation(all_dates,oscar_mode)
      



if __name__ == "__main__":
    main()
