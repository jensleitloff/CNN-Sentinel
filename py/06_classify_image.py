#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCon 2018:
Satellite data is for everyone: insights into modern remote sensing research
with open data and Python

"""
import numpy as np
from skimage.io import imread
from skimage.util import pad
from keras.models import load_model
import gdal
from tqdm import tqdm

# input files
path_to_image = "../data/karlsruhe/2018_zugeschnitten.tif"
path_to_model = "../data/models/vgg/vgg_ms_transfer_alternative_final.27-0.985.hdf5"
# output files
path_to_label_image = "../data/karlsruhe/2018_zugeschnitten_10m_vgg_ms_label.tif"
path_to_prob_image = "../data/karlsruhe/2018_zugeschnitten_10m_vgg_ms_prob.tif"

# read image and model
image = np.array(imread(path_to_image), dtype=float)
_, num_cols_unpadded, _ = image.shape
model = load_model(path_to_model)
# get input shape of model
_, input_rows, input_cols, input_channels = model.layers[0].input_shape
_, output_classes = model.layers[-1].output_shape
in_rows_half = int(input_rows/2)
in_cols_half = int(input_cols/2)

# read image and model
image = np.array(imread(path_to_image), dtype=float)
_, num_cols_unpadded, _ = image.shape
model = load_model(path_to_model)
# get input shape of model
_, input_rows, input_cols, input_channels = model.layers[0].input_shape
_, output_classes = model.layers[-1].output_shape
in_rows_half = int(input_rows/2)
in_cols_half = int(input_cols/2)

# import correct preprocessing
if input_channels is 3:
    from image_functions import preprocessing_image_rgb as preprocessing_image
else:
    from image_functions import preprocessing_image_ms as preprocessing_image

# pad image
image = pad(image, ((input_rows, input_rows),
                    (input_cols, input_cols),
                    (0, 0)), 'symmetric')

# don't forget to preprocess
image = preprocessing_image(image)
num_rows, num_cols, _ = image.shape

# sliding window over image
image_classified_prob = np.zeros((num_rows, num_cols, output_classes))
row_images = np.zeros((num_cols_unpadded, input_rows,
                       input_cols, input_channels))
for row in tqdm(range(input_rows, num_rows-input_rows), desc="Processing..."):
    # get all images along one row
    for idx, col in enumerate(range(input_cols, num_cols-input_cols)):
        # cut smal image patch
        row_images[idx, ...] = image[row-in_rows_half:row+in_rows_half,
                                     col-in_cols_half:col+in_cols_half, :]
    # classify images
    row_classified = model.predict(row_images, batch_size=1024, verbose=0)
    # put them to final image
    image_classified_prob[row, input_cols:num_cols-input_cols, : ] = row_classified

# crop padded final image
image_classified_prob = image_classified_prob[input_rows:num_rows-input_rows,
                                              input_cols:num_cols-input_cols, :]
image_classified_label = np.argmax(image_classified_prob, axis=-1)
image_classified_prob = np.sort(image_classified_prob, axis=-1)[..., -1]

# write image as Geotiff for correct georeferencing
# read geotransformation
image = gdal.Open(path_to_image, gdal.GA_ReadOnly)
geotransform = image.GetGeoTransform()

# create image driver
driver = gdal.GetDriverByName('GTiff')
# create destination for label file
file = driver.Create(path_to_label_image,
                     image_classified_label.shape[1],
                     image_classified_label.shape[0],
                     1,
                     gdal.GDT_Byte,
                     ['TFW=YES', 'NUM_THREADS=1'])
file.SetGeoTransform(geotransform)
file.SetProjection(image.GetProjection())
# write label file
file.GetRasterBand(1).WriteArray(image_classified_label)
file = None
# create destination for probability file
file = driver.Create(path_to_prob_image,
                     image_classified_prob.shape[1],
                     image_classified_prob.shape[0],
                     1,
                     gdal.GDT_Float32,
                     ['TFW=YES', 'NUM_THREADS=1'])
file.SetGeoTransform(geotransform)
file.SetProjection(image.GetProjection())
# write label file
file.GetRasterBand(1).WriteArray(image_classified_prob)
file = None
image = None
