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


# Because the files may not be in a chronological order, we sort them so that the timeseries data we are creating will be in a chronological order.
def get_start_date_from_SNODAS_filename(s):
    datetime_str = s.split('_')[-1].split('.')[0] # format is YYYYMMDD
    datetime_object = datetime.strptime(datetime_str, '%Y%m%d')
    return datetime_object

######


input_folder = '/data0/images/SNODAS/skagit/'
zarr_output_path = '/data0/images/SNODAS/skagit/SNODAS_test.zarr'

# given a shapefile, get the bounding box coords and subset dataset to that area
sf_path = Path('/home/spestana/git/Skagit/raw_data/gis/SkagitRiver_BasinBoundary.shp').expanduser()
sf = gpd.read_file(str(sf_path))
#minx, miny, maxx, maxy = sf.geometry[0].bounds

if __name__ == '__main__':
    with dask_start_cluster(
                            workers=6,
                            threads=2,
                            ip_address='http://dshydro.ce.washington.edu',
                            port=":8786",
                            open_browser=False,
                            verbose=True,
                            ) as client:


        # get all the original SNODAS netcdf files
        nc_files = sorted(
            glob.glob(os.path.join(input_folder, '*.nc')),
            key=get_start_date_from_SNODAS_filename
        )


        # open one file to get the grid we'll match all other files to (thank you Eric Gagliano)
        ds_reference = xr.open_dataset(nc_files[-1])

        files = nc_files
        datetimes = [get_start_date_from_SNODAS_filename(s) for s in files]

        for i,file in enumerate(files):
            print(f"Processing {i+1} of {len(files)}...", end="\r")
            new_file_name = file.replace(
                    "/skagit/",
                    "/skagit/withtime/",
                )
            if not os.path.exists(new_file_name):
                try:
                    ds = xr.open_dataset(file, decode_coords="all")
                    ds = ds.assign_coords({"time": datetimes[i]})
                    ds = ds.expand_dims("time")
                    ds = ds.reset_coords(drop=True)
                    ds = ds.rename({'Band1': 'swe'}) # Rename Band1 as a more indicative name: swe
                    ds = ds.interp_like(ds_reference, method='linear')
                    da = ds['swe']
                    da = da.rio.write_crs('EPSG:4326')
                    da.to_netcdf(new_file_name)
                except Exception as err:
                    print(f"Failed on {file}")
                    print(f"Error: {err}")

        # now get the new SNODAS netcdf files we just created
        nc_files = sorted(
            glob.glob(os.path.join('/data0/images/SNODAS/skagit/withtime/', '*.nc')),
            key=get_start_date_from_SNODAS_filename
        )
        
        # Open multifile dataset
        # Open all the raster files as a single dataset (combining them together)
        # Why did we choose chunks = 500? 100MB?
        # https://docs.xarray.dev/en/stable/user-guide/dask.html#optimization-tips
        ds = xr.open_mfdataset(nc_files, chunks={'time': 100}, ) #preprocess=fix_grid  combine='nested', concat_dim='time',
        
        # Dask's rechunk documentation: https://docs.dask.org/en/stable/generated/dask.array.rechunk.html

        # 0:-1 specifies that we want the dataset to be chunked along the 0th dimension -- the time dimension, which means that each chunk will have all 40 thousand values in time dimension
        # 1:'auto', 2:'auto' and balance=True specifies that dask can freely rechunk along the latitude and longitude dimensions to attain blocks that have a uniform size
        ds['swe'].data.rechunk(
            {0:-1, 1:'auto', 2:'auto'}, 
            block_size_limit=1e8, 
            balance=True
        )
        
        # Assign the dimensions of a chunk to variables to use for encoding afterwards
        t,y,x = ds['swe'].data.chunks[0][0], ds['swe'].data.chunks[1][0], ds['swe'].data.chunks[2][0]
        
        ds['swe'].encoding = {'chunks': (t, y, x)}

        # clip to watershed area
        ds_clipped = ds.rio.write_crs('epsg:4326')
        ds_clipped = ds_clipped.rio.clip(sf.geometry, sf.crs, drop=True)
        
        #ds_clipped['swe'].encoding = {'chunks': (t, y, x)}
        
        # Create an output zarr file and write these chunks to disk
        # if already exists, remove it here
        if os.path.exists(zarr_output_path):
            shutil.rmtree(zarr_output_path, ignore_errors=False)
            
        del ds_clipped.swe.attrs['grid_mapping'] # for some reason, I have to remove this
        ds_clipped.to_zarr(zarr_output_path)
        
        # Display 
        source_group = zarr.open(zarr_output_path)
        source_array = source_group['swe']
        print(source_group.tree())
        print(source_array.info)
        del source_group
        del source_array
