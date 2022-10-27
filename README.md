# ti-DEM
A high performance objected-oriented Discrete Element Method (DEM) simulator in [Taichi](https://github.com/taichi-dev/taichi). 
The coupling scheme named [ti-DEMPM](https://github.com/Yihao-Shi/ti-DEMPM)
- developed by Shi-Yihao Zhejiang Universiy(In Progress) 

## Examples
### The angle of slope (linear contact model)
<p align="center">
  <img src="https://github.com/Yihao-Shi/TaichiDEM/blob/version-updated/result3_1.gif" width="50%" height="50%" />
</p>
$\theta = 24$

### The angle of slope (linear rolling contact model)
<p align="center">
  <img src="https://github.com/Yihao-Shi/TaichiDEM/blob/version-updated/result4_1.gif" width="50%" height="50%" />
</p>
$\theta = 40$

### demo
<p align="center">
  <img src="https://github.com/Yihao-Shi/TaichiDEM/blob/version-updated/result.gif" width="50%" height="50%" />

## Features
### Discrete Element Method 
  - only spherical particles supported)

  - Search Algorithm
    1. Sorted based
    2. Multilevel linked cell

## Future Work
  1. Multisphere particles (clump) 
  2. GPU memory allocate
  4. Pre-processing (i.e., input obj files and JSON)

## Install
1. Install essential dependencies
```
bash requirements.sh
```
2. Set up environment variables
```
sudo gedit ~/.bashrc
```
3. Add the installation path to bashrc file:
```
export tiDEMPM=/user_path/tiDEMPM
```
4. Run the test
```
python test.py
```

## Acknowledgememt
Implementation is largely inspired by [ComFluSoM](https://github.com/peizhang-cn/ComFluSoM).
