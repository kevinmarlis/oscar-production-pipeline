import numpy as np
import xarray as xr
import math
from ..config.setup import *


def interpolate_dataset(ds):
     if SSH_MODE == "neurost":
        lon = np.arange(0, 360, 0.1)
        lat = np.arange(-89.9, 90.0, 0.1)
     elif SSH_MODE  == "cmems":
        lon = np.arange(0, 360, 0.125)
        lat = np.arange(-89.9375, 90.0625, 0.125)
     else:
        raise ValueError(f"Unknown SSH source '{SSH_MODE}'. Expected 'neurost' or 'cmems'.")
     

     ds_interp = ds.interp(longitude=lon, latitude=lat)

     return ds_interp


def calculate_gradient(ds, var):
        # Degrees to meters conversion factor
        R_earth = 6381e3  # in meters
        ddegdm = 180.0 / (math.pi * R_earth)

        # X gradient (longitude)
        dsx = ds[[var]].differentiate("longitude")
        dsx = dsx.rename({var: var + 'x'})
        dsx[var + 'x'] *= ddegdm / np.cos(np.deg2rad(ds['latitude']))

        # Y gradient (latitude)
        dsy = ds[[var]].differentiate("latitude")
        dsy = dsy.rename({var: var + 'y'})
        dsy[var + 'y'] *= ddegdm

        # Merge gradients into single dataset
        dsgr = xr.merge([dsx, dsy])

        ds_shape = ("time", "latitude", "longitude")
        dsgr = dsgr.transpose(*[dim for dim in ds_shape if dim in dsgr.dims])

        return dsgr
