# load libraries
import netCDF4 as nc
import xarray as xr
import glob
import pandas as pd
import os
from itertools import zip_longest
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cf
import ruptures as rpt
from datetime import datetime

# Flatten 3D array of Closeness Centrality into 1D array with shape (n_samples,)
def flatten(data):
    flat_data = []
    for i in range(data.shape[2]):
        for j in range(data.shape[1]):
            for k in range(data.shape[0]):
                flat_data.append(data[k][j][i])
    return np.array(flat_data)

# Get the 3D index of the change points
def get_3d_index(arr, var_shape):
    index = []
    for i in range(len(arr)-1):
        ind = np.unravel_index(arr[i], var_shape)
        index.append(ind)
    return index

# Return the time, lat, and lon values related to the change points
def get_coords(index, ds):
    time = []
    lat = []
    lon = []
    for i in range(len(index)):
        t = ds['time'].values[index[i][0]]
        t = t.astype('datetime64[D]').astype(str)
        la = ds['lat'].values[index[i][1]]
        lo = ds['lon'].values[index[i][2]]
        time.append(t)
        lat.append(la)
        lon.append(lo)
    return time, lat, lon

# CC data 2003
file_pattern = '../../private/complex_network_coefficients/2000-2009_run_20240105_1808/rasterfiles/Europe/2003/CN_Europe_0.25x0.25deg_CC_2003-06*.nc'
file_paths = glob.glob(file_pattern)

# Open and concatenate files
cc_2003_06 = xr.open_mfdataset(file_paths)

# Rename coefficient
cc_2003_06 = cc_2003_06.rename({'coefficient': 'CC'})

# # Save the merged data to a new NetCDF file
# if os.path.exists("cc_2003_06.nc"):
#     os.remove("cc_2003_06.nc")
#     cc_2003.to_netcdf("cc_2003_06.nc")
# else:
#     cc_2003.to_netcdf("cc_2003_06.nc")

# data = nc.Dataset('cc_2003_06.nc')
variable = cc_2003_06['CC']
input = flatten(np.array(variable))

bkps_rbf = rpt.BottomUp('rbf').fit_predict(input, n_bkps=4)

print(bkps_rbf)