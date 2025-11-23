# 📘 Computer Vision Notes (Beginner → Advanced)

## 1. Introduction to Computer Vision

Computer Vision (CV) enables computers to interpret visual information
from images and videos.

### Goals of CV

-   Image classification
-   Object detection
-   Segmentation
-   Recognition & tracking
-   Image generation
-   3D reconstruction

### Difference: Image Processing vs Computer Vision

  Image Processing         Computer Vision
  ------------------------ ------------------------------
  Low-level operations     High-level understanding
  No learning              Usually ML/DL based
  Example: blur, sharpen   Example: classify cat vs dog

## 2. Image Basics

### Pixel

Smallest unit of an image.

### Types of images

-   Grayscale (0--255)
-   RGB (3-channel)
-   Binary (0 or 1)
-   RGBA (alpha channel)

### Image as a matrix

-   Grayscale → 2D
-   RGB → 3D (H × W × 3)

## 3. Color Models

-   RGB
-   BGR (OpenCV)
-   HSV
-   YCrCb / LAB

## 4. Image Transformations

### Geometric

-   Translation
-   Rotation
-   Scaling \### Photometric
-   Brightness
-   Contrast

## 5. Image Filtering

-   Mean filter
-   Gaussian
-   Median
-   Bilateral
-   Canny edge detection

## 6. Morphology

-   Dilation
-   Erosion
-   Opening
-   Closing

## 7. Thresholding

-   Simple
-   Adaptive
-   Otsu

## 8. Contours

Used to detect shapes and boundaries.

## 9. Feature Detection

-   Harris
-   SIFT
-   SURF
-   ORB

## 10. Traditional Object Detection

-   HOG
-   Haar Cascades

## 11. Deep Learning for CV

CNN = Convolution + Pooling + FC layers.

## 12. CNN Architectures

-   LeNet
-   AlexNet
-   VGG
-   ResNet
-   Inception
-   MobileNet

## 13. Object Detection (DL)

-   R-CNN
-   Fast/Faster R-CNN
-   YOLO
-   SSD
-   RetinaNet

## 14. Segmentation

-   UNet
-   SegNet
-   Mask R-CNN

## 15. Face Recognition

-   MTCNN
-   FaceNet
-   Dlib

## 16. Video Analytics

-   Optical flow
-   Tracking (KCF, MOSSE, CSRT)

## 17. Generative Models

-   Autoencoders
-   GANs
-   Diffusion models

## 18. 3D CV

-   Depth estimation
-   SLAM

## 19. OpenCV Functions

-   imread
-   imshow
-   resize
-   rotate
-   Canny
-   threshold
-   findContours

## 20. Project Ideas

-   Document scanner
-   Lane detection
-   Real-time YOLO
-   Pose estimation
