# image-processing
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
- Lower IoU threshold (0.5) aligns with PASCAL VOC standard evaluation
- Confusion matrix showed strong per-class detection across all 5 vowels
- PR curve analysis used to determine optimal confidence threshold

---

## Skills Demonstrated
MATLAB · Deep Learning · Transfer Learning · Object Detection · 
Data Augmentation · Model Evaluation · Computer Vision

---

## Certificate
MathWorks — Object Detection with Deep Learning
Completed: March 2026
