# DL3DTO - Deep-Learning-for-3D-Topology-Optimization
Contains codes associated with our paper "DL3DTO - MATLAB/Python codes for 3D Topology optimization with Deep Learning"
The codes contains following files:

DL Data Generation Matlab codes

dataGeneration32_bulk.m - Main file for Bulk modulus data generation. Requires mat files initialDes.mat and vfrmin.mat. Calls code Top3D_maxbulk.m
dataGeneration32_shear.m - Main file for shear modulus data generation. Requires mat files initialDes.mat and vfrmin.mat.Calls code Top3D_maxshear.m
Top3D_maxbulk.m - topology optimization code for objective function of maximum Bulk modulus.Developed by [1] slightly modified.
Top3D_maxshear.m - topology optimization code for objective function of maximum Shear modulus.Developed by [1] slightly modified.

Data file Generation Matlab codes
GenerateVoxel_main - Main program to create wireframe of isosurface of TPMS and generate voxel. Calls subroutine GenerateVoxel.m.
GenerateVoxel.m - Developed by [2]. 
display_3D.m - displays figure of any 3D voxelized structure.

MATLAB Data files
initialDes.mat - contains initial topology density of a Gyroid with 32x32x32 voxels of volume fraction ~58%.Generated from GenerateVoxel_main.m
vfrmin.mat - contains the volume fraction (Vf) in the range of 25%-45% and filter radius (rmin) values in range of 1.2- 2.5 cm for each datapoint, total 2751 rows.

Deep Learning Python code
gyr_ResUnet.py - reads the input data genarated from MATLAB codes above (bulk.txt and shear.txt) from path_Data which has to be modified before running. 

Output files
From MATLAB code
bulk.txt - contains 2751 rows, each row having 1 for bulk identifier, Vf, rmin, topology optimized densities, flattened 6x6 constitutive matrix
shear.txt - contains 2751 rows, each row having 2 for shear identifier, Vf, rmin, topology optimized densities, flattened 6x6 constitutive matrix
From Python code
y_gt_data - Ground truth topology optimized density 32x32x32. 
y_pred_data - DL predicted topology optimized density 32x32x32.
x_ty_data - Identifier of datapoint(bulk/shear)
x_mf_data - Vf of datapoint
x_rm_data - rmin of datapoint


Code References (Please cite these reference papers while using code from this repository)
[1] J. Gao, H. Li, L. Gao, and M. Xiao, “Topological shape optimization of 3D micro-structured materials using energy-based homogenization method,” Adv. Eng. Softw., vol. 116, pp. 89–102, Feb. 2018, doi: 10.1016/j.advengsoft.2017.12.002.
[2] G. Dong, Y. Tang, and Y. F. Zhao, “A 149 Line Homogenization Code for Three-Dimensional Cellular Materials Written in matlab,” J. Eng. Mater. Technol., vol. 141, no. 1, Jul. 2018, doi: 10.1115/1.4040555.
