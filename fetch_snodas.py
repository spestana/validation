import os
import glob
import shutil
import numpy as np
import regionmask 
import xarray as xr
import rioxarray
import zarr
from datetime import datetime, timedelta
from dask.distributed import Client, LocalCluster
from pathlib import Path
import geopandas as gpd
from validation import SNODAS, MountainHub, Elevation, utils as ut

# given a shapefile, get the bounding box coords and subset dataset to that area
sf_path = Path('/home/spestana/git/Skagit/raw_data/gis/SkagitRiver_BasinBoundary.shp').expanduser()
sf = gpd.read_file(str(sf_path))
minx, miny, maxx, maxy = sf.geometry[0].bounds

start_date = datetime(2003,10,1)
end_date = datetime(2011,10,1)
ndays = end_date - start_date


# Fetch data from SNODAS
print(f'Fetching {ndays.days} days of SNODAS from {start_date} to {end_date}')
for n, date in enumerate(start_date + timedelta(n) for n in range(ndays.days)):
    output_path = date.strftime('/data0/images/SNODAS/skagit/SNODAS_%Y%m%d.nc')
    if not os.path.exists(output_path):
        print(f'Downloading day {n+1} of {ndays.days} to {output_path}', end='\r')
        try:
            # download file
            snodas_ds = SNODAS.snodas_ds(date, code=1034) # 1034 for SWE
            ut.save_netcdf(snodas_ds, output_path)
            # crop to bounds
            ds = xr.open_dataset(output_path)
            ds = ds.sel(lat=slice(miny,maxy), lon=slice(minx,maxx))
            ds.to_netcdf(output_path)
            ds.close()
        except Exception as err:
            print(f"\nError: {err}")

