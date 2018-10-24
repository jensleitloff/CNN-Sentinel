# Satellite data is for everyone: insights into modern remote sensing research with open data and Python

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
(or use tensorflow version, i.e. `from tensorflow import keras`)

See also:
* Keras: https://keras.io/

## Links
* DeepHyperX - Deep learning for Hyperspectral imagery: https://gitlab.inria.fr/naudeber/DeepHyperX/


## How to get Sentinel-2 data
1. Register at Copernicus [Open Access Hub](https://scihub.copernicus.eu/dhus/#/home) or [EarthExplorer[(https://earthexplorer.usgs.gov/)]
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


## Datasets

**This talk:**
* EuroSAT Data ([Link](http://madm.dfki.de/downloads))


**Platforms for datasets:**

- HyperLabelMe: a Web Platform for Benchmarking Remote Sensing Image Classifiers ([Link](http://hyperlabelme.uv.es/))
- GRSS Data and Algorithm Standard Evaluation (DASE) website ([Link](http://dase.ticinumaerospace.com/))


**Datasets:**

- UC Merced Land Use Dataset ([Link](http://weegee.vision.ucmerced.edu/datasets/landuse.html))
- AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification ([Link](https://captain-whu.github.io/AID/))
- NWPU-RESISC45 (RGB, [Link](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html))
- Zurich Summer Dataset (RGB, [Link](https://sites.google.com/site/michelevolpiresearch/data/zurich-dataset))



