
# VisualOccupancyDetectionSystem
This computer vision project is aimed at finding unoccupied spaces in a parking lot

## Dataset
The dataset consists of around 126,000 positive and negative images.
- Dataset : https://drive.google.com/file/d/1sFkz9FN_zE81jw50PIXtBnJtxlMHoDk3/view?usp=sharing
- Original dataset : CNRPark-EXT http://cnrpark.it/

## Training

- parking_bw.py - Converts the original images to 32x32 BnW images and then trains the CNN model.
- parking_color.py - Uses 32x32x3 images to train the CNN model.

In our testing, we found out that using BnW images doesn't affect the accuracy and conspicuously faster than using color images.
## Deployment
- Import the video file into the Deployment folder.
- run takeScreenShot.py 
- run SelectROIs.py and select the locations of parking spaces by dragging the mouse pointer.
- run Detect.py
(Currently works only on prerecorded videos)
