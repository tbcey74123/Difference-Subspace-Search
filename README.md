# Difference-Subspace-Search
This is the official implementation of Human-in-the-Loop Differential Subspace Search in High Dimensional Latent Space [SIGGRAPH 2020].

## Pretrained Models

The pretrained weights of the nerual networks are not maintained here.  
Please download the weights from  
https://drive.google.com/file/d/1nm-tU-jZm5AQbxJlx2L-UvlUNx4bc6G-/view?usp=sharing  
and put the `pretrained_weights` folder to `[path/to/this/repositoy]/` before running the codes using neural network

## Environment Settings

This project is developed on Windows OS, so it is not guaranteed to be executable on other operating systems.  
The core requirements across this project is listed below.

* python 3.6.1
* numpy (`pip install numpy`)
* tensorflow (`pip install tensorflow==1.14` or `pip install tensorflow-gpu==1.14`)
  * Tensorflow 2 is not supported so please make sure the installed version is at least 1.x
  * Please refer to the official guide for setting up the gpu version of tenserflow.

## Sequential Line Search

To execute the codes of Experiments and Evaluation (User study), the SLS library must be corrected set.  
In this work, we are basically using the SLS implementation proposed in [Koyama+ 2017]. The source codes are included as a git submodule and can be found here: `[path/to/this/repositoy]/sequential-line-search`, which has an SHA1 of `9f196c144199d095ee726efad4bc6678eb46ff09`.

Its Python bindings can be installed via `pip`:
```bash
git clone https://github.com/YonghaoYue/hcimlsound.git --recursive # The "--recursive" option will clone submodules recursively
pip install ./sequential-line-search
```
Needless to say, `pip` here should be associated with the Python executable that you want use.

### Tips on Git Submodule

If you already cloned this repository but has not cloned the submodule yet, then the following command will be necessary to clone the submodule with their dependencies:
```bash
git submodule update --init --recursive
```
Ref: https://git-scm.com/docs/git-submodule

## Evaluation

  
* Sequential Line Search(SLS) python library
  * For the details, please refer to the SLS subsection.

### Required for GANSynth

* librosa (`pip install librosa`)

### Required for Graphics UI

* PyQt5 (`pip install pyqt5`)
* matplotlib (`pip install matplotlib`)

## Sequential Line Search


