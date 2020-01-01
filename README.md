# Separability and Geometry of Object Manifolds in Deep Neural Networks


[![bioRxiv shield](https://img.shields.io/badge/bioRxiv-644658-red.svg?style=flat)](https://www.biorxiv.org/content/10.1101/644658v2)
[![DOI](https://img.shields.io/badge/DOI-https://doi.org/10.1101/644658-blue.svg?style=flat)](https://doi.org/10.1101/644658)


## Contents

- [Abstract](#abstract)
- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Citation](#citation)
- [License](./LICENSE)
- [Issues](https://github.com/ebridge2/lol/issues)

# Abstract

Stimuli are represented in the brain by the collective population responses of sensory neurons, and an object presented under varying conditions gives rise to a collection of neural population responses called an *object manifold*. Changes in the object representation along a hierarchical sensory system are associated with changes in the geometry of those manifolds, and recent theoretical progress connects this geometry with *classification capacity*, a quantitative measure of the ability to support object classification. Deep neural networks trained on object classification tasks are a natural testbed for the applicability of this relation. We show how classification capacity improves along the hierarchies of deep neural networks with different architectures. We demonstrate that changes in the geometry of the associated object manifolds underlie this improved capacity, and shed light on the functional roles different levels in the hierarchy play to achieve it, through orchestrated reduction of manifolds’ radius, dimensionality and inter-manifold correlations.

# Overview

This repository provides *Matlab implementation* of the algorithms described in the paper, allowing for 
 * direct estimation of classification capacity;
 * numerical estimation of object manifolds geometry (i.e. manifolds radius, dimension and inter-manifold correlations) and the classification capacity predicated by our mean-field theory.

Furthermore, we provide the code used to generate smooth manifolds described in the paper, 
as well as the code used in the analysis of both point-cloud and smooth manifolds.


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

## Install dnn-object-manifolds

 * Clone or download current from project page in github or from git command line:
```
git clone https://github.com/sompolinsky-lab/dnn-object-manifolds.git
```
 * Below we denote the folder where this project is now located as `<ROOT>`.
 
## Install CPLEX

 * Download CPLEX Studio from IBM's website; it is available for free for academic use, as described [here](https://developer.ibm.com/docloud/blog/2019/07/04/cplex-optimization-studio-for-students-and-academics/) and [here](https://optimiser.com/cplex-free-for-academic-use/);
 * Unpack or install CPLEX;
 * Locate the installation directory and add it to the matlab path (e.g. for Windows add
 `C:\Program Files\IBM\ILOG\CPLEX_Studio126\cplex\matlab\x64_win64`).
 
## Install MatConvNet

 * Download MatConvNet from [the official website](http://www.vlfeat.org/matconvnet/);
 * Unpack it under the root folder of the current project and name it `MatConvNet`;
 * Install it as described in [Installing and compiling the library] (http://www.vlfeat.org/matconvnet/install/)
 * Download the models you wish to use:
  ```
 cd <ROOT>/MatConvNet
 mkdir models
 cd models
 wget http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat
 wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat
 wget http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat
 ```

# Demo

## Smooth manifolds


## Point-cloud manifolds


# Citation

Please cite our pre-print at bioRxiv:
```
@article{cohen2019separability,
  title={Separability and Geometry of Object Manifolds in Deep Neural Networks},
  author={Cohen, Uri and Chung, SueYeon and Lee, Daniel D and Sompolinsky, Haim},
  journal={bioRxiv},
  pages={644658},
  year={2019},
  publisher={Cold Spring Harbor Laboratory}
}
```
