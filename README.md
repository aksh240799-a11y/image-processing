
# Object Detection with Transfer Learning

## Overview
Fine-tuned a YOLOX object detector using transfer learning to detect 
American Sign Language (ASL) vowels (A, E, I, O, U) in MATLAB.

---

## Pipeline
Data Preparation → Transfer Learning → Training → Evaluation → Augmentation

---

## Technical Details
- **Model:** YOLOX (tiny-coco pretrained)
- **Framework:** MATLAB Deep Learning Toolbox
- **Dataset:** ASL Vowels (Ground Truth — 200 images)
- **Split:** 70% Train / 10% Validation / 20% Test
- **Optimizer:** Adam
- **Epochs:** 100
- **Input Size:** 224 x 224 x 3

---

## Key Concepts Implemented
- Transfer learning on pretrained YOLOX (COCO → custom ASL classes)
- Ground truth data preparation and bbox datastore creation
- Class balance verification before training
- Data augmentation — horizontal flip with bbox spatial transformation
- Model evaluation — IoU threshold, confusion matrix, precision-recall curves

---

## Results
- Evaluated at IoU threshold 0.5 and 0.85
- Lower IoU threshold (0.5) improves the classification detection for each class
- Confusion matrix showed strong per-class detection across all 5 vowels
- PR curve analysis used to determine optimal confidence threshold
- Transferring Bounding box values along with GT labels for precise object detection for augmented datasets

---

## Skills Demonstrated
MATLAB · Deep Learning · Transfer Learning · Object Detection · 
Data Augmentation · Model Evaluation · Computer Vision

---

## Visual Results : 
1. Confusion Matrix with IoU (Intersection over Union) = 0.85
   <img width="523" height="395" alt="image" src="https://github.com/user-attachments/assets/5cdb86ad-1e9c-4605-a9a9-c02451b3f608" />

2. Confusion Matrix with IoU (Intersection over Union) = 0.5
   <img width="338" height="215" alt="image" src="https://github.com/user-attachments/assets/ce68425a-0277-4dff-b0e9-c0e0a02ad647" />

3. ASL Detection through transfer learning :
   <img width="315" height="215" alt="image" src="https://github.com/user-attachments/assets/3e667354-3e99-474c-82c6-09b27006f1c3" />

4. Significance of Data Augmentation : 
   a. Augmenting only GT. labels
   <img width="636" height="211" alt="image" src="https://github.com/user-attachments/assets/ac65f116-ccbc-4308-96b8-9310806966e5" />
   b. Augmenting Bounding box value along with GT labels : 
   <img width="441" height="297" alt="image" src="https://github.com/user-attachments/assets/f56f8966-6645-498b-ab13-d035555d8973" />


## Certificate
MathWorks — Object Detection with Deep Learning
<img width="886" height="570" alt="image" src="https://github.com/user-attachments/assets/a3f3a404-475a-4cfb-876c-e819f048f6aa" />

Completed: March 2026
