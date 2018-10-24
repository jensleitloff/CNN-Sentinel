# -*- coding: utf-8 -*-

"""Some statistics about the EuroSAT dataset."""

import glob
from osgeo import gdal
import numpy as np
import os

def getMeanStd(path, n_bands=3, n_max=-1):
    """Get mean and standard deviation from images.

    Parameters
    ----------
    path : str
        Path to training images
    n_bands : int
        Number of spectral bands (3 for RGB, 13 for Sentinel-2)
    n_max : int
        Maximum number of iterations (-1 = all)

    Return
    ------

    """
    if not os.path.isdir(path):
        print("Error: Directory does not exist.")
        return 0
    
    mean_array = [[] for _ in range(n_bands)]
    std_array = [[] for _ in range(n_bands)]

    # iterate over the images
    i = 0
    for tif in glob.glob(path+"*/*.*"):
        if (i < n_max) or (n_max == -1):
            ds = gdal.Open(tif)
            for band in range(n_bands):
                mean_array[band].append(
                    np.mean(ds.GetRasterBand(band+1).ReadAsArray()))
                std_array[band].append(
                    np.std(ds.GetRasterBand(band+1).ReadAsArray()))
            i+=1
        else:
            break

    # results
    res_mean = [np.mean(mean_array[band]) for band in range(n_bands)]
    res_std = [np.mean(std_array[band]) for band in range(n_bands)]

    # print results table
    print("Band |   Mean   |   Std")
    print("-"*28)
    for band in range(n_bands):
        print("{band:4d} | {mean:8.3f} | {std:8.3f}".format(
            band=band, mean=res_mean[band], std=res_std[band]))
    
    return res_mean, res_std

if __name__ == "__main__":
    getMeanStd(path="data/PyCon/RGB/train/", n_bands=3)