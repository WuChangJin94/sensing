# Sensing
 
## Installation
Instructions for installing the project. Include any prerequisites, steps, and commands necessary for setting up the project locally.
First clone the following repository.
### First clone the following repository
```bash
git clone git@github.com:WuChangJin94/sensing.git
```
### Change directory into sensing folder
```bash
cd sensing
```
### Clone the GMFlow repository
```bash
git clone https://github.com/WuChangJin94/unimatch.git
```

## Build Docker
Instructions for Docker image building for the project.
### Change directory into Docker folder and build the image
```bash
cd ~/sensing/Docker/gpu
```
### Build the image
```bash
. build.sh
```
## To inference
### Start the Docker with
docker_run.sh or gpu_run.sh
### Join the Docker with
docker_join.sh or gpu_join.sh
### Start jupyter notebook with
colab_jupyter.sh
### Goes to the notebooks directory from jupyter notebook browser to run
UniMatch_Setup.ipynb
### Place the video of interest in
/unimatch/demo/
### Perform inference with 
Optical Flow.ipynb
### See output folder for result
/../demo/output/
