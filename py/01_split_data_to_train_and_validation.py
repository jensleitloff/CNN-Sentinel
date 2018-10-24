# -*- coding: utf-8 -*-
"""
PyCon 2018:
Satellite data is for everyone: insights into modern remote sensing research
with open data and Python

"""
import os
import random
import shutil

random.seed(42)
# variables
# root path to folders "AnnualCrop, Forest ..." in home ("~")
path_to_all_images = "~/Documents/Data/EuroSAT/AllBands"
path_to_all_images = r'C:\Users\Jens\Downloads\EuroSAT_RGB'
# path to new created folders "train" and "validation" with subfolders
# "AnnualCrop, Forest ..." in home ("~")
path_to_split_datasets = "~/Documents/Data/PyCon/AllBands"
path_to_split_datasets = r'C:\Users\Jens\Downloads\PyCon\RGB'
# percentage of validation data (between 0 an 1)
percentage_validation = 0.3
# !!! If "True", complete "path_to_split_datasets" tree will be deleted !!!
delete_old_path_to_split_datasets = True

# contruct path
path_to_home = os.path.expanduser("~")
path_to_all_images = path_to_all_images.replace("~", path_to_home)
path_to_split_datasets = path_to_split_datasets.replace("~", path_to_home)
# create directories if necessary
if delete_old_path_to_split_datasets and os.path.isdir(path_to_split_datasets):
    shutil.rmtree(path_to_split_datasets)
path_to_train = os.path.join(path_to_split_datasets, "train")
path_to_validation = os.path.join(path_to_split_datasets, "validation")
if not os.path.isdir(path_to_train):
    os.makedirs(path_to_train)
if not os.path.isdir(path_to_validation):
    os.makedirs(path_to_validation)

# copy files
sub_dirs = [sub_dir for sub_dir in os.listdir(path_to_all_images)
            if os.path.isdir(os.path.join(path_to_all_images, sub_dir))]
for sub_dir in sub_dirs:
    # list and shuffle images in class directories
    current_dir = os.path.join(path_to_all_images, sub_dir)
    files = os.listdir(current_dir)
    random.shuffle(files)
    # split files into train and validation set
    split_idx = int(len(files)*percentage_validation)
    files_for_validation = files[:split_idx]
    files_for_train = files[split_idx:]
    # copy files to path_to_split_datasets
    if not os.path.isdir(os.path.join(path_to_train, sub_dir)):
        os.makedirs(os.path.join(path_to_train, sub_dir))
    if not os.path.isdir(os.path.join(path_to_validation, sub_dir)):
        os.makedirs(os.path.join(path_to_validation, sub_dir))
    for file in files_for_train:
        shutil.copy2(os.path.join(current_dir, file),
                     os.path.join(path_to_train, sub_dir))
    for file in files_for_validation:
        shutil.copy2(os.path.join(current_dir, file),
                     os.path.join(path_to_validation, sub_dir))
