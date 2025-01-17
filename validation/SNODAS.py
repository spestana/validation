import re
import os
import tarfile
import gzip
from datetime import datetime, timedelta
from io import BytesIO
from osgeo import gdal, gdal_array, osr

import numpy as np
import xarray as xr
import validation.utils as ut

def snodas_url(date, masked=None):
    """Get url of SNODAS data for given date.
    
    There are two versions of SNODAS, a masked version and an unmasked version. 
    The masked data files represent snow cover in the contiguous United States, extending into Canada for certain drainage basins.
    The unmasked data files represent snow cover in the contiguous United States, in addition to extending well into Canada as well as outlines the coast and contains parts of Mexico.

    Keyword arguments:
    date -- Date to fetch SNODAS data for
    """
    if masked == True: # masked data available from September 30th 2003 to the present
        print("you chose masked", date)
        return date.strftime('ftp://sidads.colorado.edu/DATASETS/NOAA/G02158/masked/%Y/%m_%b/SNODAS_%Y%m%d.tar')
    elif masked == False: # unmasked data available from December 9th 2009 to the present
        print("you chose unmasked", date)
        return date.strftime('ftp://sidads.colorado.edu/DATASETS/NOAA/G02158/unmasked/%Y/%m_%b/SNODAS_unmasked_%Y%m%d.tar')
    elif masked == None: # default behaviour switches from masked to unmasked on January 1st 2010
        if date >= datetime(2003,9,30) and date < datetime(2010,1,1):
            print("auto pick masked", date)
            return date.strftime('ftp://sidads.colorado.edu/DATASETS/NOAA/G02158/masked/%Y/%m_%b/SNODAS_%Y%m%d.tar')
        elif date >= datetime(2010,1,1):
            print("auto pick unmasked", date)
            return date.strftime('ftp://sidads.colorado.edu/DATASETS/NOAA/G02158/unmasked/%Y/%m_%b/SNODAS_unmasked_%Y%m%d.tar')


def snodas_file_format(date):
    """Get format string for gzipped SNODAS files for given date.

    Keyword arguments:
    date -- Date to fetch SNODAS data for
    """
    if date >= datetime(2003,9,30) and date < datetime(2010,1,1):
        return date.strftime('us_ssmv1%%itS__T0001TTNATS%Y%m%d05HP001.%%s.gz')
    elif date >= datetime(2010,1,1):
        return date.strftime('zz_ssmv1%%itS__T0001TTNATS%Y%m%d05HP001.%%s.gz')



# Remove lines longer than 256 characters from header (GDAL requirement)
def clean_header(hdr):
    """Remove lines longer than 256 characters from header (GDAL requirement)."""
    new_hdr = BytesIO()
    for line in hdr:
        if len(line) <= 256:
            new_hdr.write(line)

    # Cleanup
    new_hdr.write(b'')
    new_hdr.seek(0)
    hdr.close()

    return new_hdr

def clean_paths(paths, tar):
    """Corrects paths of files in tar."""
    new_paths = []
    for path in paths:
        try:
            info = tar.getmember(path)
        except:
            path = './' + path
        new_paths.append(path)
    return new_paths

def tar_to_snodas(tar, gz_format, code=1036):
    """Converts tar archive to GDAL dataset.

    Keyword arguments:
    tar -- tar object
    gz_format -- format for gzipped files in archive
    code -- SNODAS product code (default 1036 [Snow Depth])
    """

    extensions = ['dat', 'txt']
    # Untar and extract files
    gz_paths = [gz_format % (code, extension) for extension in extensions]
    vsi_paths = ['/vsimem/' + path[:-3] for path in gz_paths]

    # Some paths in tar file have ./ preceeding, some do not
    # Use clean_tar_paths to find and use correct paths
    gz_paths = clean_paths(gz_paths, tar)

    gz_files = [tar.extractfile(path) for path in gz_paths]
    dat_file, hdr_file = [gzip.GzipFile(fileobj=file, mode='r') for file in gz_files]

    # Read data into buffers
    hdr_file = clean_header(hdr_file)
    dat = dat_file.read()
    hdr = hdr_file.read()

    # Convert to GDAL Dataset
    gdal.FileFromMemBuffer(vsi_paths[0], dat)
    gdal.FileFromMemBuffer(vsi_paths[1], hdr)
    ds = gdal.Open(vsi_paths[1])

    # Close / Unlink Virtual Files
    tar.close()
    dat_file.close()
    hdr_file.close()
    gdal.Unlink(vsi_paths[0])
    gdal.Unlink(vsi_paths[1])

    return ds

def snodas_ds(date, masked=None, code=1036):
    """Get SNODAS data as GDAL dataset for specific date.

    Keyword arguments:
    date -- datetime object
    code -- integer specifying SNODAS product (default 1036 [Snow Depth])
            Other options:  1025: Precipitation
                            1034: Snow water equivalent
                            1036: Snow depth
                            1038: Snow pack average temperature
                            1039: Blowing snow sublimation
                            1044: Snow melt
                            1050: Snow pack sublimation
    
    """
    url = snodas_url(date, masked)
    gz_format = snodas_file_format(date)
    tar = ut.url_to_tar(url)
    return tar_to_snodas(tar, gz_format, code=code)
