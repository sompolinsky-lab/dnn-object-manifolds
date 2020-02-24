# Separability and Geometry of Object Manifolds in Deep Neural Networks


[![bioRxiv shield](https://img.shields.io/badge/bioRxiv-644658-red.svg?style=flat)](https://www.biorxiv.org/content/10.1101/644658v3)
[![bioRxiv shield](https://img.shields.io/badge/bioRxiv-Supplemenrary-green.svg?style=flat)](https://www.biorxiv.org/content/biorxiv/early/2020/02/17/644658/DC1/embed/media-1.pdf?download=true)
[![DOI](https://img.shields.io/badge/DOI-10.1038/s41467--020--14578--5-blue.svg?style=flat)](https://doi.org/10.1038/s41467-020-14578-5)


## Contents

- [Abstract](#abstract)
- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [How to use with your data](#how-to-use-with-your-data)
- [Citation](#citation)
- [License](./LICENSE)
- [Issues](https://github.com/sompolinsky-lab/dnn-object-manifolds/issues)

# Abstract

Stimuli are represented in the brain by the collective population responses of sensory neurons, and an object presented under varying conditions gives rise to a collection of neural population responses called an *object manifold*. Changes in the object representation along a hierarchical sensory system are associated with changes in the geometry of those manifolds, and [recent theoretical progress](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031003) connects this geometry with *classification capacity*, a quantitative measure of the ability to support object classification. Deep neural networks trained on object classification tasks are a natural testbed for the applicability of this relation. We show how classification capacity improves along the hierarchies of deep neural networks with different architectures. We demonstrate that changes in the geometry of the associated object manifolds underlie this improved capacity, and shed light on the functional roles different levels in the hierarchy play to achieve it, through orchestrated reduction of manifolds’ radius, dimensionality and inter-manifold correlations.

# Overview

This repository provides *Matlab implementation* of the algorithms described in the paper, allowing for 
 * direct estimation of classification capacity;
 * numerical estimation of object manifolds geometry (i.e. manifolds radius, dimension and inter-manifold correlations) and the classification capacity predicated by the mean-field theory based analysis used in our work.

Furthermore, we provide the code used to generate smooth manifolds described in the paper, 
as well as the code used in the analysis of both point-cloud and smooth manifolds.

For numerical estimation of object manifolds geometry and the classification capacity based on the metric used in our analysis, a follow-up work on speech recognition deep networks provides 
[a python implementation](https://github.com/schung039/neural_manifolds_replicaMFT).

# Repo Contents

- [library](./library): reusable *Matlab* library code.
- [point_cloud_analysis](./point_cloud_analysis): *Matlab* code used in the paper for the analysis of point cloud manifods.
- [smooth_manifolds_generation](./smooth_manifolds_generation): *Matlab* code used in the paper for the generation of smooth manifods.
- [smooth_manifolds_analysis](./smooth_manifolds_analysis): *Matlab* code used in the paper for the analysis of smooths manifolds.
- [FOptM](./FOptM): copy of the code of \[Wen & Yib 2010\] (*A Feasible method for Optimization with Orthogonality Constraints*) from [their web site](http://optman.blogs.rice.edu). See [their README](./FOptM/README.m) for more details.


# System Requirements

The code requires:
 * *Matlab* from MathWorks, version R2017a or later (tested on R2017a);
 * Any OS supported by the Matlab platform (tested on Windows 10, MAC OS and Gentoo Linux);
 * *CPLEX* from IBM, version 128 or later (tested on CPLEX studio 128)
 * *MatConvNet* code and publically available models from [the official website](http://www.vlfeat.org/matconvnet).

No special hardware is required, but MatConvNet may take advantage of a GPU (as indicated in its installation instuctions), which may speed-up manifold generation by a factor.


# Installation Guide

## Download dnn-object-manifolds

 * Clone or download current from project page in github or from git command line:
```
git clone https://github.com/sompolinsky-lab/dnn-object-manifolds.git
```
 * Below we denote the folder where this project is now located as `<ROOT>`.
 * Download `imagenet_all_thumbnails_64px.mat` [from figshare](https://doi.org/10.6084/m9.figshare.11494314) and save it at `<ROOT>/smooth_manifolds_generation`.
 * Typical installation time: 5 minutes
 
## Install CPLEX

 * Download CPLEX Studio from IBM's website; it is available for free for academic use, as described [here](https://developer.ibm.com/docloud/blog/2019/07/04/cplex-optimization-studio-for-students-and-academics/) and [here](https://optimiser.com/cplex-free-for-academic-use/);
 * Unpack or install CPLEX;
 * Locate the installation directory and add it to the matlab path (e.g. for Windows add
 `C:\Program Files\IBM\ILOG\CPLEX_Studio126\cplex\matlab\x64_win64`).
 * Typical installation time: 10 minutes
 
## Install MatConvNet

 * Download MatConvNet from [the official website](http://www.vlfeat.org/matconvnet/);
 * Unpack it under the root folder of the current project and name it `MatConvNet`;
 * Install it as described in [Installing and compiling the library] (http://www.vlfeat.org/matconvnet/install/); a minimal version would be as follows:
 ```
 >> mex -setup C++
 >> addpath matlab
 >> vl_compilenn
 ```
 * Download the models you wish to use using the following shell commands:
  ```
 cd <ROOT>MatConvNet
 mkdir models
 cd models
 wget http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat
 wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat
 wget http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat
 ```
 * Typical installation time: 20 minutes

# Demo

Here we demonstrate the code used in the paper; to run it on your own data [see below](#how-to-use-with-your-data).

## Smooth 1-d manifolds

Open *Matlab* and change current folder to `<ROOT>`.
```
chdir('smooth_manifolds_generation')
```

Stimuli initialization code:
```
init_imagenet;
```

Network initialization code (should be done only once):
```
NETWORK_TYPE = 1; % 1: alexnet, 3: resnet50, 5: vgg16
generate_convnet_model_metadata(NETWORK_TYPE);
% Copy to analysis directory
copyfile('convnet_alexnet_model.mat', '../smooth_manifolds_analysis') % 'alexnet', 'resnet50' or 'vgg16'
```

Generate smooth 1-d manifolds- 7 affine transformations (40 minutes):
```
range_factor=0.5;
N_OBJECTS=128;
N_SAMPLES=15;
% distributed generation of objects
for id=1:28; generate_convnet_one_dimensional_change(range_factor, N_OBJECTS, N_SAMPLES, NETWORK_TYPE, id); end 
% collect object representations
generate_convnet_one_dimensional_change(range_factor, N_OBJECTS, N_SAMPLES, NETWORK_TYPE, 1:28);
```

Move to analysis folder:
```
copyfile('alexnet', '../smooth_manifolds_analysis'); % 'alexnet', 'resnet50' or 'vgg16'
chdir('../smooth_manifolds_analysis')
```

Direct estimation of classification capacity (25 minutes):
``` 
layer_number = 20; % e.g. for alexnet 1: pixel layer, 20: feature layer
check_convnet_capacity_one_dimensional_change2(N_OBJECTS, range_factor, N_SAMPLES, NETWORK_TYPE, 0,  (layer_number-1)*7+1);
```

Numerical estimation of object manifolds geometry using mean-field theory (120 minutes):
```
check_convnet_covariance_low_rank_approx_optimal_K(N_OBJECTS, range_factor, N_SAMPLES, NETWORK_TYPE, layer_number, 5, 1);
```

## Smooth 2-d manifolds

Open *Matlab* and change current folder to `<ROOT>`.
```
chdir('smooth_manifolds_generation')
```

Stimuli initialization code:
```
init_imagenet;
```

Network initialization code (should be done only once):
```
NETWORK_TYPE = 1; % 1: alexnet, 3: resnet50, 5: vgg16
generate_convnet_model_metadata(NETWORK_TYPE);
% Copy to analysis directory
copyfile('convnet_alexnet_model.mat', '../smooth_manifolds_analysis') % 'alexnet', 'resnet50' or 'vgg16'
```

Generate smooth 2-d manifolds: 2 affine transformations (100 minutes):
```
show_imagenet_random_change(128, 16, 2, 0);

range_factor=0.5;
N_OBJECTS=64;
N_SAMPLES=201;
n_batches = 4;

% distributed generation of objects
for id=1:n_batches*2; generate_convnet_random_change(N_OBJECTS, range_factor, N_SAMPLES, NETWORK_TYPE, 2, n_batches, id); end
% collect object representations
generate_convnet_random_change(N_OBJECTS, range_factor, N_SAMPLES, NETWORK_TYPE, 2, n_batches, 1:n_batches*2);
```

Move to analysis folder
```
copyfile('alexnet', '../smooth_manifolds_analysis'); % 'alexnet', 'resnet50' or 'vgg16'
chdir('../smooth_manifolds_analysis')
```

Direct estimation of classification capacity (15 minutes):
``` 
layer_number = 20; % e.g. for alexnet 1: pixel layer, 20: feature layer
check_convnet_capacity_random_change2(N_OBJECTS, range_factor, N_SAMPLES, NETWORK_TYPE, 2, 0, '', layer_number);
```

Numerical estimation of object manifolds geometry using mean-field theory (120 minutes):
```
check_convnet_covariance_low_rank_approx_optimal_K(N_OBJECTS, range_factor, N_SAMPLES, NETWORK_TYPE, layer_number, 5, 2);
```

## Point-cloud manifolds

A similar implementation is also [availabe in Python in another repositoy](https://github.com/schung039/neural_manifolds_replicaMFT).

# How to use with your data

### Preparation
The easiest way to use the above method with your own data is to generate file(s) with the following format:

 1. Choose a name for your data-set (e.g. convnet name, experiment details) in a *Matlab* string named `name_prefix`;
 2. Prepare a *Matlab* variable named `tuning_function` of class `double` with dimensions of `[N_SESSIONS, N_NEURONS, N_SAMPLES, N_OBJECTS]` or `[N_NEURONS, N_SAMPLES, N_OBJECTS]` when `N_SESSIONS=1`;
 3. Prepare a *Matlab* variable named `data_titles` of class `cell` with dimensions of `[N_SESSIONS, 1]` where each entry is a string describing the sessions (e.g. layer name, brain region name);
 4. Save them all:
```
save(sprintf('%s_tuning.mat', name_prefix), 'tuning_function', 'data_titles', '-v7.3');
```

Notes:
 * When the number of neurons differ accross different sessions, `N_NEURONS` should be the maximal value and `NaN` values should be used for missing neurons.
 * When the number of samples differ accross different objects, `N_SAMPLES` should be the maximal value and `NaN` values should be used for missing samples.
 
### Direct estimation of capacity
You can perform direct estimation of classification capacity:
```
session_ids = 0; % 0- all sessions, otherwise: specific session
check_capacity(name_prefix, session_ids);
```
This would results in the creation of a file named `<name_prefix>_capacity*.mat` with a variable named `capacity_results` of size `N_SESSIONS` who contains the critical number of neurons found:
```
file=matfile(name)
Ac=N_OBJECTS./capacity_results;
```

### Numerical estimation of capacity and object manifolds geometry
You can perform numerical estimation of object manifolds geometry and the predicated classification capacity:
```
session_ids = 0; % 0- repeat for all sessions, otherwise: only the given session
n_neurons = 0; % 0- use all neurons, otherwise: repeat multiple times sampling the given number of neurons
check_covariance_low_rank_approx_optimal_K(name_prefix, session_ids, n_neurons)
```
This would result in the creation of a file named `<name_prefix>_lowrank_optimalK*.mat` with a variables such that:
```
file=matfile(name);
Ac = file.theory_capacity_results % predicated classification capacity 
Rm = mean(sqrt(file.mean_argmax_norm2_results),2) % manifolds radii
Dm = mean(file.effective_dimension_results),2) % manifolds dimensions
```

# Citation

Please cite our *Nature Communications* paper:
```
@article{cohen2020separability,
  title={Separability and geometry of object manifolds in deep neural networks},
  author={Cohen, Uri and Chung, SueYeon and Lee, Daniel D and Sompolinsky, Haim},
  journal={Nature Communications},
  volume={11},
  number={1},
  pages={1--13},
  year={2020},
  publisher={Nature Publishing Group}
}
```
