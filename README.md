[![DOI](https://zenodo.org/badge/154472226.svg)](https://zenodo.org/badge/latestdoi/154472226)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Analyzing Sentinel-2 satellite data in Python with Keras and TensorFlow

Overview about state-of-the-art land-use classification from satellite data with CNNs based on an open dataset

## Outline

* [Scripts you will find here](#scripts-you-will-find-here)
* [Requirements (what we used):](#requirements--what-we-used--)
* [Setup Environment](#setup-environment)
* [Our talks about this topic](#our-talks-about-this-topic)
* [Resources](#resources)
* [How to get Sentinel-2 data](#how-to-get-sentinel-2-data)

## Scripts you will find here

* `01_split_data_to_train_and_validation.py`: split complete dataset into train and validation
* `02_train_rgb_finetuning.py`: train VGG16 or DenseNet201 using RGB data with pre-trained weights on ImageNet
* `03_train_rgb_from_scratch.py`: train VGG16 or DenseNet201 from scratch using RGB data
* `04_train_ms_finetuning.py`: train VGG16 or DenseNet201 using multisprectral data with pre-trained weights on ImageNet
* `04_train_ms_finetuning_alternative.py`: an alternative way to train VGG16 or DenseNet201 using multisprectral data with pre-trained weights on ImageNet
* `05_train_ms_from_scratch.py`: train VGG16 or DenseNet201 from scratch using multisprectral data
* `06_classify_image.py`: a simple implementation to classify images with trained models
* `image_functions.py`: functions for image normalization and a simple generator for training data augmentation
* `statistics.py`: a simple implementation to calculate normalization parameters (i.e. mean and std of training data)

Additionally you will find the following notebooks:

* `Image_functions.ipynb`: notebook of `image_functions.py`
* `Train_from_Scratch.ipynb`: notebook of `05_train_ms_from_scratch.py`
* `Transfer_learning.ipynb`: notebook of `02_train_rgb_finetuning.py`

## Requirements (what we used)

* python 3.6.6
* tensorflow-gpu (1.11) with Cuda Toolkit 9.0
* keras (2.2.4)
* scikit-image (0.14.1)
* gdal (2.2.4) for `06_classify_image.py`

## Setup Environment

Append conda-forge to your Anaconda channels:

```bash
conda config --append channels conda-forge
```

Create new environment:

```bash
conda create -n pycon scikit-image gdal tqdm
conda activate pycon
pip install tensorflow-gpu
pip install keras
```

(or use tensorflow version of keras, i.e. `from tensorflow import keras`)

See also:

* [Keras](https://keras.io/)

## Our talks about this topic

### Podcast episode @ InoTecCast

* **Title:** "Fernerkundung mit multispektralen Satellitenbildern"
* **Episode:** [Episode 18](https://inoteccast.de/18-fernerkundung-mit-multispektralen-satellitenbildern/)
* **Podcast:** [InoTecCast](https://inoteccast.de)
* **Language:** German
* **Date:** July 2019

### M3 Minds mastering machines 2019 @ Mannheim

* **Title:** "Satellite Computer Vision mit Keras und Tensorflow - Best practices und beispiele aus der Forschung"
* **Slides:** [Slides](slides/M3-2019_RieseLeitloff_SatelliteCV.pdf)
* **Language:** German
* **Date:** 15 - 16 May 2019
* **URL:** [m3-konferenz.de](https://www.m3-konferenz.de)
* **Abstract:**

> Im Forschungsfeld des Maschinellen Lernens werden zunehmend leicht zugängliche Framework wie Keras, Tensorflow oder Pytorch verwendet. Hierdurch ist ein Austausch und die Wiederverwendung bestehender (trainierter) neuronaler Netze möglich.
>
> Wir am Institut für Photogrammetrie und Fernerkundung (IPF) des Karlsruher Institut für Technologie (KIT) beschäftigen uns unter anderem mit der Analyse von optischen Satellitendaten. Satellitenprogramme wie Sentinel-2 von Copernicus liefern wöchentliche, weltweite und dabei frei zugängliche multispektrale Bilder, die eine Vielzahl neuartiger Anwendungen ermöglichen. Wir nehmen das zum Anlass, eine interaktive Einführung in die Auswertung dieser Satellitendaten mit Learnings aus unserer täglichen Forschung zu geben. Wir sprechen unter anderem über die folgenden Themen:
>
> * Einfacher Umgang mit georeferenzierten Bilddaten
> * Einführung in Learning-From-Scratch und Transfer Learning mit Keras
> * Anpassung von fertigen Netzen an neue Eingangsdaten (RGB → multispektral)
> * Anschauliche Interpretation von Klassifikationsergebnissen
> * Best Practices aus unserer Forschung, die die Arbeit mit Neuronalen Netzen wesentlich vereinfachen und beschleunigen
> * Code und Daten für die ersten Schritte mit CNNs mit Keras in Python, welche in einem GitHub Repository zur Verfügung gestellt werden

### PyCon.DE 2018 @ Karlsruhe

* **Title:** "Satellite data is for everyone: insights into modern remote sensing research with open data and Python"
* **Slides:** [Slides](slides/PyCon2018_LeitloffRiese_SatelliteData.pdf)
* **Video:** [youtube.com/watch?v=tKRoMcBeWjQ](https://www.youtube.com/watch?v=tKRoMcBeWjQ)
* **Language:** English
* **Date:** 24 - 28 October 2018
* **URL:** [de.pycon.org](https://de.pycon.org)
* **Abstract:**

> The largest earth observation programme Copernicus (http://copernicus.eu) makes it possible to perform terrestrial observations providing data for all kinds of purposes. One important objective is to monitor the land-use and land-cover changes with the Sentinel-2 satellite mission. These satellites measure the sun reflectance on the earth surface with multispectral cameras (13 channels between 440 nm to 2190 nm). Machine learning techniques like convolutional neural networks (CNN) are able to learn the link between the satellite image (spectrum) and the ground truth (land use class). In this talk, we give an overview about the state-of-the-art land-use classification with CNNs based on an open dataset.
>
> We use different out-of-box CNNs for the Keras deep learning library (https://keras.io/). All networks are either included in Keras itself or are available from Github repositories. We show the process of transfer learning for the RGB datasets. Furthermore, the minimal changes required to apply commonly used CNNs to multispectral data are demonstrated. Thus, the interested audience will be able to perform their own classification of remote sensing data within a very short time. Results of different network structures are visually compared. Especially the differences of transfer learning and learning from scratch are demonstrated. This also includes the amount of necessary training epochs, progress of training and validation error and visual comparison of the results of the trained networks. Finally, we give a quick overview about the current research topics including recurrent neural networks for spatio-temporal land-use classification and further applications of multi- and hyperspectral data, e.g. for the estimation of water parameters and soil characteristics.

## Resources

**This talk:**

* EuroSAT Data (Sentinel-2, [Link](http://madm.dfki.de/downloads))

**Platforms for datasets:**

* HyperLabelMe: a Web Platform for Benchmarking Remote Sensing Image Classifiers ([Link](http://hyperlabelme.uv.es/))
* GRSS Data and Algorithm Standard Evaluation (DASE) website ([Link](http://dase.ticinumaerospace.com/))

**Datasets:**

* ISPRS 2D labeling challenge ([Link](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html))
* UC Merced Land Use Dataset ([Link](http://weegee.vision.ucmerced.edu/datasets/landuse.html))
* AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification ([Link](https://captain-whu.github.io/AID/))
* NWPU-RESISC45 (RGB, [Link](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html))
* Zurich Summer Dataset (RGB, [Link](https://sites.google.com/site/michelevolpiresearch/data/zurich-dataset))
* **Note**: Many German state authorities offer free geodata (high resolution images, land use/cover vector data, ...) over their geoportals. You can find an overview of all geoportals here ([geoportals](https://www.geoportal.nrw/geoportale_bundeslaender_nachbarstaaten))

**Image Segmentation Resources:**

* More than 100 combinations for image segmentation routines with Keras and pretrained weights for endcoding phase ([Segmentation Models](https://github.com/qubvel/segmentation_models))
* Another source for image segmentation with Keras including pretrained weights ([Keras-FCN](https://github.com/aurora95/Keras-FCN))
* Great link collection of image segmantation networks and datasets ([Link](https://github.com/mrgloom/awesome-semantic-segmentation))
* Free land use vector data of NRW ([BasisDLM](https://www.bezreg-koeln.nrw.de/brk_internet/geobasis/landschaftsmodelle/basis_dlm/index.html) or [openNRW](https://open.nrw/en/node/154))

**Other:**

* DeepHyperX - Deep learning for Hyperspectral imagery: [gitlab.inria.fr/naudeber/DeepHyperX/](https://gitlab.inria.fr/naudeber/DeepHyperX/)

## How to get Sentinel-2 data

1. Register at Copernicus [Open Access Hub](https://scihub.copernicus.eu/dhus/#/home) or [EarthExplorer](https://earthexplorer.usgs.gov/)
2. Find your region
3. Choose tile(s) (→ area) and date
    * Less tiles makes things easier
    * Less clouds in the image are better
    * Consider multiple dates for classes like “annual crop”
4. Download L1C data
5. Decide of you want to apply L2A atmospheric corrections
    * Your CNN might be able to do this by itself
    * If you want to correct, use [Sen2Cor](http://step.esa.int/main/third-party-plugins-2/sen2cor/)
6. Have fun with the data
