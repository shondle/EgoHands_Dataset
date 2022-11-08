# EgoHands_Dataset

This project allows you to query over 48 hours of Google Glass video complex first-person interactions from the EgoHands Dataset using filters (location, activity, viewer, partner) to create a PyTorch Database object of respective images and black-and-white hand segmentation labels. All using Python!

In addition, query bounding boxes, segmentations masks, and base images by frame, and filter through videos as stated.

## Code Overview
*Before running the code, please refer to the next section which goes over how to set the environment*.

Each file contains a description of what it does. 

*getMetaBy.py, getSegmentationMask.py, getFramePath.py, getBoundingBoxes.py, and DEMO1.py* contain (for the most part) the same descriptions as from the original EgoHands MATLAB code. The rest (*getTrainingImgs.py, visualizeDataset.py, dataset.py*) are commented out by myself.

To get a quick overview of what this project can do, run DEMO1.py and read the commented out code. To view a sample PyTorch dataset queried from the videos, run visualizedDataset.py. All methods used and referenced include descriptions in the files themeselves.

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

In your IDE/Compiler, where the project is downloaded, set the interpreter to the new environment you just created. 

### Downloading the EgoHands frames from Indiana University
