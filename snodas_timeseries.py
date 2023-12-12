import os
import glob
import shutil
import numpy as np
import xarray as xr
import rioxarray
import zarr
from datetime import datetime, timedelta
from dask.distributed import Client, LocalCluster
import geopandas as gpd
from pathlib import Path
from metloom.pointdata import SnotelPointData
from metloom.variables import SensorDescription, SnotelVariables



def dask_start_cluster(
    workers,
    threads=1,
    ip_address=None,
    port=":8787",
    open_browser=False,
    verbose=True,
):
    """
    Starts a dask cluster. Can provide a custom IP or URL to view the progress dashboard.
    This may be necessary if working on a remote machine.
    """
    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=threads,
        #silence_logs=logging.ERROR,
        dashboard_address=port,
    )

    client = Client(cluster)

    if ip_address:
        if ip_address[-1] == "/":
            ip_address = ip_address[:-1]  # remove trailing '/' in case it exists
        port = str(cluster.dashboard_link.split(":")[-1])
        url = ":".join([ip_address, port])
        if verbose:
            print("\n" + "Dask dashboard at:", url)
    else:
        if verbose:
            print("\n" + "Dask dashboard at:", cluster.dashboard_link)
        url = cluster.dashboard_link

    if port not in url:
        if verbose:
            print("Port", port, "already occupied")

    if verbose:
        print("Workers:", workers)
        print("Threads per worker:", threads, "\n")

    if open_browser:
        webbrowser.open(url, new=0, autoraise=True)

    return client




if __name__ == '__main__':
    
    
    # given a shapefile, get the bounding box coords and subset dataset to that area
    sf_path = Path('/home/spestana/git/Skagit/raw_data/gis/SkagitRiver_BasinBoundary.shp').expanduser()
    sf = gpd.read_file(str(sf_path))

    # find all SNOTEL sites within basin
    variables = [SnotelPointData.ALLOWED_VARIABLES.SNOWDEPTH,
        SnotelPointData.ALLOWED_VARIABLES.SWE,
        SnotelPointData.ALLOWED_VARIABLES.PRECIPITATION,
        SnotelPointData.ALLOWED_VARIABLES.TEMP,
        SnotelPointData.ALLOWED_VARIABLES.RH,]

    # Find all the points in the area for our variables
    points = SnotelPointData.points_from_geometry(sf, variables)
    print(f'Found {len(points)} SNOTEL sites within shapefile boundary')
    # turn that iterator into a dataframe
    pts_df = points.to_dataframe()
    
    
    
    with dask_start_cluster(workers=6, threads=2, verbose=False) as client:
        
        ds_clipped = xr.open_zarr('/data0/images/SNODAS/skagit/SNODAS_skagit.zarr')
        
        # get basin-wide stats
        ds_mean_swe = ds_clipped.swe.mean(axis=(1,2)) / 1000 # scale factor of 1000 to get swe in meters
        ds_median_swe = ds_clipped.swe.median(axis=(1,2)) / 1000 # scale factor of 1000 to get swe in meters
        ds_q25_swe = ds_clipped.swe.quantile(0.25, dim=['lon','lat']) / 1000 # scale factor of 1000 to get swe in meters
        ds_q75_swe = ds_clipped.swe.quantile(0.75, dim=['lon','lat']) / 1000 # scale factor of 1000 to get swe in meters
        # output mean swe basin-wide        
        df_mean_swe = ds_mean_swe.to_dataframe()
        df_mean_swe.to_csv('SNODAS_SkagitBasin_meanSWE.csv')
        
        
        # for each SNOTEL site, output a timeseries from SNODAS at that site
        for i, point in pts_df.iterrows():
            ds_pt = ds_clipped.sel(lon=point.geometry.x, lat=point.geometry.y, method='nearest').swe / 1000 # scale factor of 1000 to get swe in meters
            df_pt = ds_pt.to_dataframe()
            out_filename = f'SNODAS_at_{point['name'].replace(' ','')}_{point['id'].replace(':','_')}.csv'
            df_pt.to_csv(out_filename)