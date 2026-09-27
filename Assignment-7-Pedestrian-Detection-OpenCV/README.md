# Assignment 7: Pedestrian Detection using OpenCV

This project uses OpenCV's HOG descriptor with the default pre-trained SVM people detector to find pedestrians in street images.  
Detections are compared against the Penn-Fudan ground truth boxes using Intersection over Union, and precision, recall, and F1-score are reported across different detector settings.

## Dataset

The Penn-Fudan Pedestrian dataset is not included here because of its size. Download and extract it into this folder before running the notebook:

- https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip

Expected folder layout after extraction:

```
PennFudanPed/PNGImages
PennFudanPed/Annotation
```

## Note on OpenCV version

`cv2.HOGDescriptor` was removed in OpenCV 5. This notebook requires OpenCV 4.x:

```
pip install "opencv-python<5"
```
