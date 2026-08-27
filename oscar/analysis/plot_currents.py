import os
import matplotlib.pyplot as plt
import numpy as np 
from ..utils.data_utils import extract_date
from ..config.setup import REGION


def plot_interp(data_array, title, filename, save_dir, var, vmin, vmax):

    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.subplots(1, 1, figsize=(6, 4)) 
    plt.title(title)

    p = plt.pcolormesh(data_array.longitude, data_array.latitude, data_array[var].squeeze(), shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(p)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,filename), dpi=100, bbox_inches='tight')
    plt.close()
    

def plot_grad(data_array, title, filename, save_dir, varx, vary, vmin, vmax):

    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)) 
    fig.suptitle(title)

    p = axes[0].pcolormesh(data_array.longitude, data_array.latitude, data_array[varx].squeeze(), shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(p, ax=axes[0])
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    p = axes[1].pcolormesh(data_array.longitude, data_array.latitude, data_array[vary].squeeze(), shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(p, ax=axes[1])
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,filename), dpi=100, bbox_inches='tight')
    plt.close()

    

def extract_data(ds):
    u = ds['u'].isel(time=0).where(ds['u'] != -999.)
    v = ds['v'].isel(time=0).where(ds['v'] != -999.)
    lon = ds['lon']
    lat = ds['lat']
    
    lon_adj = lon.where(lon <= 180, lon - 360)  
    #lon_adj = ((lon + 180) % 360) - 180
    if REGION == 'global':
        lon_adj = lon_adj.sortby(lon_adj)
    
    region_bounds = {
        'global':      {'lon_min': -180, 'lon_max': 180,  'lat_min': -90,  'lat_max': 90},
        'gulf_stream': {'lon_min': -85,  'lon_max': -60,  'lat_min': 25,   'lat_max': 45},
        'caribbean':  {'lon_min': -90,  'lon_max': -55,  'lat_min': 9,    'lat_max': 23},
        'indian_ocean':     {'lon_min': 20,   'lon_max': 120,  'lat_min': -50,  'lat_max': 30},
        'australia_nz': {'lon_min': 110,'lon_max': 180,'lat_min': -50,'lat_max': -10}
    }

    if REGION not in region_bounds:
        raise ValueError(f"Unknown region '{REGION}'. Choose from: {list(region_bounds.keys())}")

    bounds = region_bounds[REGION]

    lon_mask = (lon_adj >= bounds['lon_min']) & (lon_adj <= bounds['lon_max'])
    lat_mask = (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])



    u_sub = u.where(lon_mask & lat_mask, drop=True).squeeze()
    v_sub = v.where(lon_mask & lat_mask, drop=True).squeeze()
    lon_sub = lon_adj.where(lon_mask, drop=True)
   
    lat_sub = lat.where(lat_mask, drop=True)
    speed_sub = np.sqrt(u_sub**2 + v_sub**2)
    lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)

  
    return u_sub, v_sub, lon_sub, lat_sub, speed_sub, lon2d, lat2d


def plot_currents(ds_path, filename, lon2d, lat2d, u, v, speed, title, ssh_mode, vmin=0, vmax=1.6):
    year, month = extract_date(ds_path)

    plt.figure(figsize=(12, 8))

    if REGION == 'global':
        skip = 10
    else:
        skip =2

    # Plot speed as background
    p = plt.pcolormesh(lon2d, lat2d, speed.T, shading='auto', cmap='turbo', vmin=vmin, vmax=vmax)
    plt.colorbar(p, label='m/s')

    # Plot current vectors (quiver)
    plt.quiver(
        lon2d[::skip, ::skip],
        lat2d[::skip, ::skip],
        u.T[::skip, ::skip],
        v.T[::skip, ::skip],
        scale=None,
        color='black',
        width=0.001
    )

    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()