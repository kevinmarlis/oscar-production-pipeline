import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import os
from glob import glob
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import linregress


def get_unique_drifter_daily_avg(ds, output_folder): 
    ''' 
    Input: month of 6 hourly drifters in one netcdf 
    Output: individual drifter daily average points for the year 
    ''' 
    os.makedirs(output_folder, exist_ok=True) 
    
    unique_ids = np.unique(ds['ID'].values) 
    
    for drifter_id in unique_ids: 
        drifter_ds = ds.where(ds['ID'] == drifter_id, drop=True) 
        drifter_ds = drifter_ds.where((drifter_ds['ve'] <= 10) & (drifter_ds['ve'] >= -10) & (drifter_ds['vn'] <= 10) & (drifter_ds['vn'] >= -10), drop=True) # there are still large values in drifters that we flag
        drifter_ds = drifter_ds.sortby('time') 
        
        try:
            daily_ds = drifter_ds.resample(time='1D').mean() 
            daily_ds['ID'] = xr.DataArray(np.full(daily_ds.dims['time'], drifter_id), dims='time') 
            outfile = os.path.join(output_folder, f"drifter_{drifter_id}_daily_avg.nc") 
            daily_ds.to_netcdf(outfile)
            
        except: #no valid data
            pass
            
        
def get_daily_avg_all_drifters(unique_drifters_dir, daily_avg_dir, selected_date, dailyfilename):

    selected_date = np.datetime64(selected_date)

    # Initialize containers
    latitudes = []
    longitudes = []
    ve_list = []
    vn_list = []
    ve_oscar_list = []
    vn_oscar_list = []
    ids = []

    # Get all .nc files in folder
    file_list = sorted(glob(os.path.join(unique_drifters_dir, "*.nc")))

    for file in file_list:
        ds = xr.open_dataset(file)

        if selected_date in ds['time']:
            try:
                ve = ds['ve'].sel(time=selected_date).values.item()
                vn = ds['vn'].sel(time=selected_date).values.item()
                lat = ds['latitude'].sel(time=selected_date).values.item()
                lon = ds['longitude'].sel(time=selected_date).values.item()
                drifter_id = ds['ID'].sel(time=selected_date).values.item()

                ve_list.append(ve)
                vn_list.append(vn)
                latitudes.append(lat)
                longitudes.append(lon)
                ids.append(str(drifter_id))

            except Exception as e:
                print(f"Skipping {file}: {e}")

    # Combine into dataset
    snapshot_ds = xr.Dataset({
        'latitude': xr.DataArray(latitudes, dims='drifter'),
        'longitude': xr.DataArray(longitudes, dims='drifter'),
        've': xr.DataArray(ve_list, dims='drifter'),
        'vn': xr.DataArray(vn_list, dims='drifter'),
        'ID': xr.DataArray(ids, dims='drifter'),
        'time': xr.DataArray([selected_date] * len(ids), dims='drifter')
    })

    snapshot_ds.to_netcdf(dailyfilename)
  

def interpolate_oscar_to_drifters(ds_oscar, ds_drifter, method='linear'):
    # Get OSCAR lat/lon and fields
    lats = ds_oscar['lat'].values
    lons = ds_oscar['lon'].values
    u = ds_oscar['u'].values
    v = ds_oscar['v'].values

    # If OSCAR has time dimension, drop it
    if u.ndim == 3:
        u = u[0, :, :]
        v = v[0, :, :]

    u = u.T  # shape now (719, 1440)
    v = v.T

    # Get drifter lat/lon
    drifter_lats = ds_drifter['latitude'].values
    drifter_lons = ds_drifter['longitude'].values

    # Handle lon wraparound if needed
    if np.any(lons > 180) and np.any(drifter_lons < 0):
        drifter_lons = (drifter_lons + 360) % 360
        
    # Set up interpolators
    u_interp = RegularGridInterpolator((lats, lons), u, method=method, bounds_error=False, fill_value=np.nan)
    v_interp = RegularGridInterpolator((lats, lons), v, method=method, bounds_error=False, fill_value=np.nan)

    # Stack into interpolation points
    interp_points = np.column_stack((drifter_lats, drifter_lons))

    # Interpolate
    u_interp_vals = u_interp(interp_points)
    v_interp_vals = v_interp(interp_points)

    ds_out = ds_drifter.copy()
    ds_out = ds_out.rename({"ve": "u_drifter", "vn": "v_drifter"})
    ds_out[f'u_oscar'] = (('drifter',), u_interp_vals)
    ds_out[f'v_oscar'] = (('drifter',), v_interp_vals)

    return ds_out



#==============DRIFTER vs OSCAR ============#
        
def plot_velocity_comparison_scatter(df, temporal_range):
    def compute_metrics(y_drifter, y_oscar):
        # Flatten and mask NaNs
        y_drifter = np.array(y_drifter).flatten()
        y_oscar = np.array(y_oscar).flatten()
        mask = ~np.isnan(y_drifter) & ~np.isnan(y_oscar)
        
        y_drifter_clean = y_drifter[mask]
        y_oscar_clean = y_oscar[mask]

        # Metrics
        difference = np.mean(y_oscar_clean - y_drifter_clean)
        rmsd = np.sqrt(np.mean((y_oscar_clean - y_drifter_clean) ** 2))
        r2 = 1 - (np.sum((y_drifter_clean - y_oscar_clean) ** 2) / 
                  np.sum((y_drifter_clean - np.mean(y_drifter_clean)) ** 2))

        return r2, rmsd, difference
    
    fig = plt.figure(figsize=(12, 6))

    # ---- Zonal ----
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(df['u_drifter'], df[f'u_oscar'], alpha=0.5, s=5)
    ax1.plot([-2, 2], [-2, 2], 'k--', linewidth=1)  # 1:1 line
    ax1.set_xlabel("Drifter Zonal Velocity (m/s)")
    ax1.set_ylabel(f"OSCAR Zonal Velocity (m/s)")
    ax1.grid(True)

    r2_u, rmsd_u, diff_u = compute_metrics(df['u_drifter'], df[f'u_oscar'])
    ax1.text(
        0.05, 0.95,
        f"R² = {r2_u:.3f}\nRMSD = {rmsd_u:.3f}\nBias = {diff_u:.3f}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
    )

    # ---- Meridional ----
    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(df['v_drifter'], df[f'v_oscar'], alpha=0.5, s=5)
    ax2.plot([-2, 2], [-2, 2], 'k--', linewidth=1)  # 1:1 line
    ax2.set_xlabel("Drifter Meridonal Velocity (m/s)")
    ax2.set_ylabel(f"OSCAR Meridonal Velocity (m/s)")
    ax2.grid(True)

    # Compute stats for meridional
    r2_v, rmsd_v, diff_v = compute_metrics(df['v_drifter'], df[f'v_oscar'])
    ax2.text(
        0.05, 0.95,
        f"R² = {r2_v:.3f}\nRMSD = {rmsd_v:.3f}\nBias = {diff_v:.3f}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
    )

    plt.suptitle(f"OSCAR vs Drifter Velocity - {temporal_range}", fontsize=14)
    plt.tight_layout()

    return fig


def compute_binned_correlations(
    data,
    components=('u', 'v'),
    lat_range=(-90, 90),
    lon_range=(-180, 180),
    bin_size=2,
    MIN_SAMPLES=5
):
    """
    Returns a dict of 2D correlation DataFrames
    """
    # Bin edges
    lat_edges = np.arange(lat_range[0], lat_range[1] + bin_size, bin_size)
    lon_edges = np.arange(lon_range[0], lon_range[1] + bin_size, bin_size)

    # Convert longitudes from 0–360 to -180–180
    lon = data["longitude"].values
    lon = ((lon + 180) % 360) - 180  # now in [-180, 180]

    # Replace the original longitude for binning
    data["longitude"] = (("drifter",), lon)

    # Compute bin indices for each drifter
    lat_idx = np.digitize(data["latitude"].values, lat_edges) - 1
    lon_idx = np.digitize(data["longitude"].values, lon_edges) - 1

    # Number of bins
    nlat = len(lat_edges) - 1
    nlon = len(lon_edges) - 1

    # Prepare arrays to hold correlation results
    corr_u_grid = np.full((nlat, nlon), np.nan)
    corr_v_grid = np.full((nlat, nlon), np.nan)

    # Compute correlation per bin
    for i in range(nlat):
        for j in range(nlon):
            mask = (lat_idx == i) & (lon_idx == j)
            if mask.sum() >= MIN_SAMPLES:  # need at least MIN_SAMPLES points
                corr_u_grid[i, j] = np.corrcoef(
                    data["u_drifter"].values[mask], data["u_oscar"].values[mask]
                )[0, 1]
                corr_v_grid[i, j] = np.corrcoef(
                    data["v_drifter"].values[mask], data["v_oscar"].values[mask]
                )[0, 1]

    # Make an xarray Dataset
    ds_corr2d = xr.Dataset(
        {
            "corr_u": (("lat", "lon"), corr_u_grid),
            "corr_v": (("lat", "lon"), corr_v_grid)
        },
        coords={
            "lat": (lat_edges[:-1] + bin_size/2),
            "lon": (lon_edges[:-1] + bin_size/2)
        }
    )

    return ds_corr2d


def plot_binned_correlation_map(
    temporal_range,
    corr_df,
    component,
    zoom_extent=None,
    cmap='jet',
    vmin=0.0,
    vmax=1.0,
    title_prefix=None
):
  
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.pcolormesh(corr_df.lon.values, corr_df.lat.values, corr_df.values, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    ax.coastlines()

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Optional zoom
    if zoom_extent is not None:
        lon_min, lon_max, lat_min, lat_max = zoom_extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
          
    title = f"Correlation {component} OSCAR vs {component} Drifters - {temporal_range}"
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label('Correlation (R)', fontsize=12)

    plt.tight_layout()
    
    return fig


def plot_density(
    df,
    temporal_range,           # 'cmems' or 'neurost'
    component: str,               # 'CMEMS' or 'NEUROST'
    xlim=(-0.6, 0.6),
    ylim=(-0.6, 0.6),
    bins=150,
    log=False,
    cmap="turbo"
):

    # drifter and oscar arrays
    x = np.asarray(df[f"{component}_drifter"].values)
    y = np.asarray(df[f"{component}_oscar"].values)

    # clean + clip
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    m = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    x = x[m]; y = y[m]

    # figure/axes
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = LogNorm() if log else None

    # density histogram
    h = ax.hist2d(x, y, bins=bins, range=[xlim, ylim], cmap=cmap, norm=norm)
    mappable = h[3]

    # y=x reference
    ax.plot(xlim, xlim, color="white", lw=1, zorder=3)

    # OLS fit (only if enough points)
    if x.size >= 3:
        slope, intercept, *_ = linregress(x, y)
        xx = np.array(xlim)
        ax.plot(xx, slope * xx + intercept, color="red", lw=2, zorder=4)
        ax.text(
            0.02, 0.95, f"SL={slope:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            color="w", fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
        )

    # labels, limits, titles
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(f"{'Log ' if log else ''}Density histogram - {temporal_range}", fontsize=14)
    ax.set_xlabel(f"Drifter {component} (m/s)")
    ax.set_ylabel(f"OSCAR {component} (m/s)")

    # colorbar
    cb = fig.colorbar(mappable, ax=ax, orientation='vertical', fraction=0.035, pad=0.04)
    cb.set_label("Counts" + (" (log scale)" if log else ""),fontsize=12)
    fig.tight_layout()

    return fig



# def make_slope_map(df_in,component,temporal_range):
#     """
#     Compute per-grid-cell OLS slope between truth_col and model_col,
#     plot the slope field, and return the figure object.
#     """
#
#     # hard-coded config
#     lat_bins = np.arange(-90, 90, 2)
#     lon_bins = np.arange(0, 361, 2)
#     MIN_SAMPLES = 5
#     vmin, vmax = 0, 1
#     cmap = 'jet'
#
#     # Copy dataset to df
#     df = df_in[['latitude','longitude',f"{component}_drifter",f"{component}_oscar"]].to_dataframe().reset_index()
#     df['longitude'] = df['longitude'] % 360
#
#     # Bin latitude/longitude
#     df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, labels=lat_bins[:-1])
#     df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, labels=lon_bins[:-1])
#
#     # Define slope helper
#     def _safe_slope(x, y):
#         x = np.asarray(x)
#         y = np.asarray(y)
#         mask = np.isfinite(x) & np.isfinite(y)
#         x = x[mask]; y = y[mask]
#         if x.size < MIN_SAMPLES or np.nanstd(x) == 0:
#             return np.nan
#         return linregress(x, y).slope
#
#     # Group by binned coordinates and compute slope
#     s = df.groupby(['lat_bin','lon_bin']).apply(lambda g: _safe_slope(g[f"{component}_drifter"], g[f"{component}_oscar"]))
#
#     # Pivot to 2D array for plotting
#     s_grid = s.reset_index().pivot(index='lat_bin', columns='lon_bin', values=0)
#
#     # ---- plot ----
#     fig = plt.figure(figsize=(12, 6))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#
#     lons = s_grid.columns.values
#     lats = s_grid.index.values
#     im = ax.pcolormesh(lons, lats, s_grid.values, vmin=vmin, vmax=vmax, cmap=cmap)
#
#     ax.coastlines()
#     ax.set_title(f"Slope of the linear regression between {component} OSCAR and {component} Drifter - {temporal_range}", fontsize=14)
#
#     gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
#     gl.top_labels = False
#     gl.right_labels = False
#
#     cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
#     cbar.set_label('Slope',fontsize=12)
#
#     plt.tight_layout()
#
#     return fig


def make_slope_map(df_in, component, temporal_range):
    """
    Compute per-grid-cell OLS slope between Drifter and OSCAR components,
    plot the slope field, and return the figure object.
    """

    # Configuration
    lat_range = (-90, 90)
    lon_range = (-180, 180)
    bin_size = 2
    MIN_SAMPLES = 5
    vmin, vmax = 0, 1
    cmap = 'jet'

    # Convert dataset to DataFrame
    df = df_in[[ 'latitude', 'longitude', f"{component}_drifter", f"{component}_oscar"]].to_dataframe().reset_index()
    
    # Wrap longitudes to [-180, 180]
    df['longitude'] = ((df['longitude'] + 180) % 360) - 180

    # Bin latitude/longitude
    lat_bins = np.arange(lat_range[0], lat_range[1] + bin_size, bin_size)
    lon_bins = np.arange(lon_range[0], lon_range[1] + bin_size, bin_size)

    df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, labels=lat_bins[:-1]).astype(float)
    df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, labels=lon_bins[:-1]).astype(float)

    # Helper to compute slope safely
    def _safe_slope(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < MIN_SAMPLES or np.nanstd(x) == 0:
            return np.nan
        return linregress(x, y).slope

    # Group by binned coordinates and compute slope
    s = df.groupby(['lat_bin', 'lon_bin']).apply(
        lambda g: _safe_slope(g[f"{component}_drifter"], g[f"{component}_oscar"])
    )

    # Pivot to 2D array
    s_grid = s.reset_index().pivot(index='lat_bin', columns='lon_bin', values=0)
    s_grid = s_grid.sort_index(axis=0).sort_index(axis=1).reindex(index=lat_bins[:-1], columns=lon_bins[:-1])
    
    # Compute edges for pcolormesh
    lon_edges = np.append(lon_bins[:-1], lon_bins[-1])
    lat_edges = np.append(lat_bins[:-1], lat_bins[-1])

    # ---- Plot ----
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.pcolormesh(lon_edges, lat_edges, s_grid.values, vmin=vmin, vmax=vmax, cmap=cmap)

    ax.coastlines()
    ax.set_title(f"Slope of {component} OSCAR vs {component} Drifter - {temporal_range}", fontsize=14)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label('Slope', fontsize=12)

    plt.tight_layout()

    return fig



def make_rms_map(df_in, component, temporal_range, vmin=0.0, vmax=0.5, zoom=None):
    """
    Compute per-grid-cell RMSD between Drifter and OSCAR components,
    plot the RMSD field, and return the figure object.
    """

    # Configuration
    lat_range = (-90, 90)
    lon_range = (-180, 180)
    bin_size = 2
    MIN_SAMPLES = 5
    cmap = 'jet'

    # Convert dataset to DataFrame
    df = df_in[['latitude','longitude',f"{component}_drifter",f"{component}_oscar"]].to_dataframe().reset_index()
    
    # Wrap longitudes to [-180, 180]
    df['longitude'] = ((df['longitude'] + 180) % 360) - 180

    # Bin latitude/longitude
    lat_bins = np.arange(lat_range[0], lat_range[1] + bin_size, bin_size)
    lon_bins = np.arange(lon_range[0], lon_range[1] + bin_size, bin_size)

    df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, labels=lat_bins[:-1]).astype(float)
    df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, labels=lon_bins[:-1]).astype(float)

    # RMSD helper
    def _safe_rmsd(drifter, oscar, nmin=MIN_SAMPLES):
        t = np.asarray(drifter)
        p = np.asarray(oscar)
        mask = np.isfinite(t) & np.isfinite(p)
        t, p = t[mask], p[mask]
        if t.size < nmin:
            return np.nan
        return float(np.sqrt(np.mean((p - t)**2)))

    # Compute RMSD per bin
    s = df.groupby(['lat_bin','lon_bin']).apply(
        lambda g: _safe_rmsd(g[f"{component}_drifter"], g[f"{component}_oscar"])
    )

    # Pivot to 2D array
    s_grid = s.reset_index().pivot(index='lat_bin', columns='lon_bin', values=0)
    
    # Reindex to full grid to avoid missing bins
    s_grid = s_grid.reindex(index=lat_bins[:-1], columns=lon_bins[:-1])

    # Compute edges for pcolormesh
    lon_edges = np.append(lon_bins[:-1], lon_bins[-1])
    lat_edges = np.append(lat_bins[:-1], lat_bins[-1])

    # ---- Plot ----
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.pcolormesh(lon_edges, lat_edges, s_grid.values, vmin=vmin, vmax=vmax,
                       cmap=cmap, shading='auto')

    ax.coastlines()
    ax.set_title(f"RMSD between {component} OSCAR and {component} Drifter - {temporal_range}", fontsize=14)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    if zoom:
        lon_min, lon_max, lat_min, lat_max = zoom
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label('RMSD (m/s)', fontsize=12)

    plt.tight_layout()
    return fig



def make_residual_corr_map(df_in, component, temporal_range, vmin=-0.2, vmax=0.2, zoom=None):
    """
    Compute per-grid-cell Pearson correlation between Drifter and OSCAR components residuals,
    plot the correlation field, and return the figure object.
    """

    # Configuration
    lat_range = (-90, 90)
    lon_range = (-180, 180)
    bin_size = 2
    MIN_SAMPLES = 5
    cmap = 'RdBu_r'

    # Convert dataset to DataFrame
    df = df_in[['latitude','longitude',f"{component}_drifter",f"{component}_oscar"]].to_dataframe().reset_index()
    
    # Wrap longitudes to [-180, 180]
    df['longitude'] = ((df['longitude'] + 180) % 360) - 180

    # Bin latitude/longitude
    lat_bins = np.arange(lat_range[0], lat_range[1] + bin_size, bin_size)
    lon_bins = np.arange(lon_range[0], lon_range[1] + bin_size, bin_size)

    df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, labels=lat_bins[:-1]).astype(float)
    df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, labels=lon_bins[:-1]).astype(float)
    
    # --- correlation helper (residual vs truth) ---
    def _safe_corr(a, b, nmin=MIN_SAMPLES):
        a = np.asarray(a)
        b = np.asarray(b)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < nmin:
            return np.nan
        a, b = a[m], b[m]
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    # --- compute residual correlation grid ---
    s = df.groupby(['lat_bin','lon_bin']).apply(lambda g: _safe_corr(g[f"{component}_drifter"], g[f"{component}_oscar"]))

    # Pivot to 2D array and reindex to full grid
    s_grid = s.reset_index().pivot(index='lat_bin', columns='lon_bin', values=0)
    s_grid = s_grid.reindex(index=lat_bins[:-1], columns=lon_bins[:-1])

    # Compute edges for pcolormesh
    lon_edges = np.append(lon_bins[:-1], lon_bins[-1])
    lat_edges = np.append(lat_bins[:-1], lat_bins[-1])

    # ---- Plot ----
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.pcolormesh(lon_edges, lat_edges, s_grid.values, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto', transform=ccrs.PlateCarree())

    ax.coastlines()
    ax.set_title(f"Pearson correlation between {component} OSCAR and {component} Drifter - {temporal_range}", fontsize=14)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    if zoom:
        lon_min, lon_max, lat_min, lat_max = zoom
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label("Pearson correlation coefficient", fontsize=12)

    plt.tight_layout()
    return fig
