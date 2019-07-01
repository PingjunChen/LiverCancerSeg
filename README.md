# Liver Tumor Segmentation - PAIP2019: Liver Cancer Segmentation

To use the code, the user needs to pre-install a few packages.
```
$ sudo apt-get install openslide-tools
$ sudo apt-get install libgeos-dev
$ pip install -r requirements.txt
```

## Preprocessing:
### 1. Download slides and unzip
Download all 50 zipped slides and two csv files, put them inside `./data/SourceData`, unzip them by running
```
$ cd preprocess
$ python unzip_slides.py
```
All slides would be unzipped into `./data/LiverImages`.

### 2. Check the segmentation masks
Visualizing the `whole` and `viable` mask of a slide can give the user an intuitive feel on how the tumor looks. Run the following code to generate the side-by-side view of the masks with the corresponding slide image.
```
$ python check_mask.py
```
Moreover, [`tissueloc`](https://github.com/PingjunChen/tissueloc) provides the algorithm to locate the boundary of real tissues in the slide. Running
```
$ python locate_tissue.py
```
The located tissue results may help in slide-level prediction stage. Both mask comparison and tissue localization results are saved in the `./data/Visualization` directory.

### 3. Check the viable tumor burden
If you want to check the provided `viable tumor burden` with the calculated result from provided masks, run the following code:
```
$ cd ../burden
$ python validate_burden.py
```

## Patch-based Slide Segmentation:
### 1. Patch sample generation
#### 1.1 viable tumor patch splitting
#### 1.2 whole tumor patch splitting

### 2. Segmentation model training
#### 2.1 Model selection
#### 2.2 Optimizer
#### 2.3 Loss function
#### 2.4 Patch Normalization

### 3. Slide tumor prediction
