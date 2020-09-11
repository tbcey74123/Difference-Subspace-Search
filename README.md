# Difference-Subspace-Search
This is the official implementation of Human-in-the-Loop Differential Subspace Search in High Dimensional Latent Space [SIGGRAPH 2020].

## Pretrained Models

The pretrained weights of the nerual networks are not maintained here.  
Please download the weights from  
https://drive.google.com/file/d/1nm-tU-jZm5AQbxJlx2L-UvlUNx4bc6G-/view?usp=sharing  
and put the `pretrained_weights` folder to `[path/to/this/repositoy]/` before running the codes using neural networks.

## Environment Settings

This project is developed on Windows OS, so it is not guaranteed to be executable on other operating systems.  
The core requirements across this project is listed below.

* python 3.6.1
* numpy (`pip install numpy`)
* tensorflow (`pip install tensorflow==1.14` or `pip install tensorflow-gpu==1.14`)
  * You may not need to install tensorflow if you are only going to run the Experiments that don't make use of neural networks.
  * Tensorflow 2 is not supported so please make sure the installed version is 1.x
  * Please refer to the official guide for setting up the gpu version of tenserflow.
  
### Other Dependencies

Please check the dependencies according to your target script.  
For example, if you are going to try our Application using GANSynth model, you must make sure the requirements for GUI & GANSynth are corrected installed.

#### Requirements for GUI

* PyQt5 (`pip install pyqt5`)

#### Requirements for MNIST & PGGAN

* OpenCV (`pip install opencv-python`)

#### Requirements for GANSynth

* librosa (`pip install librosa`)

#### Requirements for IMGAN

* PyMCubes (`pip install PyMCubes`)
  * check https://github.com/pmneila/PyMCubes

## Sequential Line Search

To execute the codes of Experiments and Evaluation (User study), the SLS library must be corrected installed.  
If you are only going to play with the Application, you can simply skip this step.  
In this work, we are basically using the SLS implementation proposed in [Koyama+ 2017]. The source codes are included as a git submodule and can be found here: `[path/to/this/repositoy]/sequential-line-search`, which has an SHA1 of `9f196c144199d095ee726efad4bc6678eb46ff09`.

Its Python bindings can be installed via `pip`:
```bash
pip install ./sequential-line-search
```

If you failed to install the library with `pip` under Windows, you can choose to manually compile it using `cmake` and put the compiled `pyd` file under `[path/to/this/repositoy]/utils`.

## Experiments

### Synthetic Experiment Local

The code to generate the result shown in the 4.2 section in our paper is `[path/to/this/repositoy]/Experiments/synthetic_experiment_local.py`.

You can execute the script by
```bash
cd Experiments
python synthetic_experiment_local.py [--test]
```
Add `--test` to have a test run to make sure the environments are corrected set.

The randomly intialized variables can be found here:  
https://drive.google.com/drive/folders/19Clma43lC55NOdHguWKxAjpuoFDEFBID?usp=sharing  
Please make sure the path is correct, i.e. the `synthetic_experiment_local` should be inside the `[path/to/this/repositoy]/Evaluation/` and the data structure should be organized as those in the drive.

### Synthetic Experiment Global

The code to generate the result shown in the 4.3 ~ 4.5 sections in our paper is `[path/to/this/repositoy]/Experiments/synthetic_experiment_global.py`.

You can execute the script by

```bash
cd Experiments
python synthetic_experiment_global.py [--test]
```
Add `--test` to have a test run to make sure the environments are corrected set.

The randomly intialized variables can be found here:  
https://drive.google.com/drive/folders/1HiEbEd6PhBUWbPwZos5Fmr9WUbsQTKY1?usp=sharing  
Please make sure the path is correct, i.e. the `synthetic_experiment_global` should be put inside the `[path/to/this/repositoy]/Evaluation/` and the data structure should be organized as those in the drive.

### MNIST

The code to generate the result shown in the 5.1 section in our paper is `[path/to/this/repositoy]/Experiments/mnist_experiment_global.py`.

You can execute the script by
```bash
cd Experiments
python mnist_experiment_global.py [--test]
```
Add `--test` to have a test run to make sure the environments are corrected set.

The randomly intialized variables can be found here:  
https://drive.google.com/file/d/1o8uCp-VAHUDnQLeW6kDkjTsgvmRH1nGe/view?usp=sharing  
Please put the extracted file `mnist_experiment_global` into `[path/to/this/repositoy]/Evaluation/`.

## Evaluation (User Study)

The code to perform the user study shown in the 5.2 section in our paper is `[path/to/this/repositoy]/Evaluation/evaluate.py`.

You can execute the script by
```bash
cd Evaluation
python evaluate.py setting_json subject_index
```
The legal option for `setting_json` is either `pggan_celebahq.json` or `gansynth.json` and the `subject_index` should be an integer.

## Application

You can play with our application by
```bash
cd Application
python search.py dataset [--output_path output_path] [--initial initial_file]
```
The legal option for `dataset` is `MNIST`, `PGGAN_celebahq`, `GANSynth`, `IMGAN_flight` and `IMGAN_chair`.  
Add `--output_path` to specify the saving path and add `--initial` to specify the inital data.  
The `initial_file` should be a text file, where the 1st line contains the latent size and the second line contains the latent variable with each dimension separated with a space.

### Application Guide

In the beginning of the application (and as well as any time during the search process), press `R` to pick a randomly selected data as the initial data. Once you decided the inital data, press the Start Button to start the search. During the search, you can always press `S` to save the current data.
