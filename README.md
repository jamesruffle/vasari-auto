# VASARI-auto
This is a codebase for automated VASARI characterisation of glioma, as detailed in the article [article](URL).

![Overview](assets/graphical_abstract.jpg)

## Table of Contents
- [What is this repository for?](#what-is-this-repository-for)
- [Usage instructions](#usage-instructions)
  -  [With tumour lesions already segmented](#with-tumour-lesions-already-segmented)
  -  [Without tumour lesions segmented yet](#without-tumour-lesions-segmented-yet)
- [Efficiency](#efficiency)
- [Usage queries](#usage-queries)
- [Citation](#citation)
- [Funding](#funding)

## What is this repository for?
VASARI...

## Usage instructions
### With tumour lesions already segmented

### Without tumour lesions segmented yet
1. Install [nnU-Net v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) | *N.B. use of a CUDA-supported GPU is strongly recommended.*
2. Download our model weights [here](https://doi.org/10.5281/zenodo.6782948).
3. Skull-strip your data. *All models have been trained to expect skull-stripped images. If not already done, there are many ways to do this, though we personally recommend [HD-BET](https://github.com/MIC-DKFZ/HD-BET).*
4. For using a specific model / sequence combinbation, see [here](#Using-a-specific-model--sequence-combination).
5. Where MRI sequence availabilty differs across the cohort, see [here](#with-variable-sequence-availability-across-your-cohort).


## Efficiency
If running VASARI-auto with lesions already segmented by your own model or annotations as detailed [here](#with-tumour-lesions-already-segmented), time to VASARI featurize is approximately **SECS**.

If pairing with our [Brain Tumour Segmentation Model]() as shown [here](#without-tumour-lesions-segmented-yet), and with GPU-accelerated hardware (prototyped on a NVIDIA GeForce RTX 3090 Ti), time to segment and VASARI-featurise per patient is approximately **10-15 seconds**.


## Usage queries
Via github issue log or email to j.ruffle@ucl.ac.uk

## Citation
If using these works, please cite the following [paper](URL):

*Citation here*

## Funding
The Medical Research Council; The Wellcome Trust; UCLH NIHR Biomedical Research Centre; Guarantors of Brain.
![funders](assets/funders.png)