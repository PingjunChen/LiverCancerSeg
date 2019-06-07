# Liver Tumor Segmentation - PAIP2019: Liver Cancer Segmentation

In order to use the following code, you need to pre-install a few dependent packages.
```
$ sudo apt-get install openslide-tools
$ sudo apt-get install libgeos-dev
$ pip install -r requirements.txt
```

## Preprocesssing:
### 1. Download slides and unzip
Download all 50 zipped slides and put them in `./data/SourceData`, unzip them by
```
$ cd preprocess
$ python unzip_slides.py
```
The slides would be unzipped into `./data/LiverImages`. To note that some manual changes need to be done to help the following analysis. First, the `viable` and `whole` mask image of `01_01_0107` is reversed, swap the names of these two files. Second, remove the `rescan` from two `01_01_0111` related files to ensure all given slides having consistent names.

### 2. Check the segmentation masks
Visualizing the `whole` and `viable` masks would give the user an intuitive feel on how these tumor looks like. Run following code to generate the side-by-side view of the masks.
```
$ python check_mask.py
```
Moreover, [`tissueloc`](https://github.com/PingjunChen/tissueloc) provides the algorithm to locate the boundary and real tissue. Running
```
$ python locate_tissue.py
```
To check the tissue localization result, which may help at the last slide-level prediction stage. Both the mask comparison and tissue localization results are saved in `./data/Visualization` directory.

### 3. Check the viable tumor burden
The format of the provided `Phase_1_tumor_burden.csv` and `Phase_2_tumor_burden.csv` is not consistent. First, change `Phase_1_tumor_burden.csv` in `./data/SourceData` to `Phase_1_tumor_burdenBK.csv`, and then standardize them by:
```
$ python adjust_phase1csv.py
```
The new generated `Phase_1_tumor_burden.csv` would have consistent format with `Phase_2_tumor_burden.csv`.  
If you want to check the provided `viable tumor burden` with the calculated result from the provided mask iamges, run following code:
```
$ cd ../burden
$ python validate_burden.py
```
You will find out there are `eight` cases that their burden values are not consistent, including `01_01_0107`.

## Patch-based Slide Segmentation:
### 1. Patch sample generation
### 2. Patch segmentation model training
### 3. Slide tumor prediction
