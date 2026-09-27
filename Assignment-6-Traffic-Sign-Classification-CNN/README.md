# Assignment 6: Traffic Sign Classification using CNN

This project trains a Convolutional Neural Network on the GTSRB dataset to recognise 43 classes of German traffic signs.  
The model is trained on 39,209 images and evaluated on the separate official test set of 12,630 images using accuracy, a confusion matrix, and per-class results.

## Dataset

The GTSRB dataset is not included in this folder because of its size. Download and extract it here before running the notebook:

- Training images: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
- Test images: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
- Test labels: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

Expected folder layout after extraction:

```
GTSRB/Final_Training/Images/00000 ... 00042
GTSRB/Final_Test/Images
GT-final_test.csv
```
