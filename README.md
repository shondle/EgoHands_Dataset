# EgoHands_Dataset

This project allows you to query over 48 hours of Google Glass video complex first-person interactions from the EgoHands Dataset using filters (location, activity, viewer, partner) to create a PyTorch Database object of respective images and black-and-white hand segmentation labels. All using Python!

In addition, query bounding boxes, segmentations masks, and base images by frame, and filter through videos as stated.

Overall, this project makes it easier for developers to run ML models for hand segmentation, and adjust their testing set, based off of the EgoHands Dataset.

## Code Overview
*Before running the code, please refer to the next section Setting Up and Running the Code*.

Each file contains a description of what it does. 

`getMetaBy.py`, `getSegmentationMask.py`, `getFramePath.py`, `getBoundingBoxes.py`, and `DEMO1.py` contain (for the most part) the same descriptions as from the original EgoHands MATLAB code. The rest (`getTrainingImgs.py`, `visualizeDataset.py`, `dataset.py`) are commented out by myself.

To get a quick overview of what this project can do, run `DEMO1.py` and read the commented out code. To view a sample PyTorch dataset queried from the videos, run `visualizeData.py`. All methods used and referenced include descriptions in the files themeselves.

## Setting Up and Running the Code

### Package Manager and Required Libraries
Use a package manager, such as Anaconda, to download the following libraries in a new environment.
- SciPy
- NumPy
- Pandas
- PyTorch
- OpenCV
- Matplotlib
- pathlib

Enter the following in the Anaconda console, 
```console
conda create -n EgoHandsDataset
````
Next, 
```console
conda activate EgoHands Dataset
conda install pip
pip install ipykernel
python -m ipykernel install — user --name EgoHandsDataset --display-name “EgoHandsDataset"

conda install pup
pip install opencv-python

//Scipy, numpy, pandas, openCV, matplotlib, pathlib
conda install -c conda-forge scipy numpy pandas matplotlib pathlib
// Pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
````



In your IDE/Compiler, where the project is downloaded, set the interpreter to the new environment you just created. To find the path, follow [this link](https://docs.anaconda.com/anaconda/user-guide/tasks/integration/python-path/)

```console
//Mac
which python
//windows
where python
### Downloading the EgoHands frames from Indiana University
Go to [this link](http://vision.soic.indiana.edu/projects/egohands/) and download the "Labelled Data" Zip Archive (which should be around 1.3GB). Unzip this file and drag the _LABELLEDSAMPLES_
