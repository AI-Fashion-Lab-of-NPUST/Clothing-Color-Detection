# Clothing-Color Analysis Pipeline

A computer vision pipeline for detecting and identifying clothing colors from torso images extracted from CCTV footage.

## Overview

The pipeline combines clothing segmentation, color clustering, and color assignment to identify the dominant colors of clothing while minimizing the influence of background pixels and unrelated objects.

## Workflow

```text
Torso Image
     ↓
SegFormer-B2
(Clothing Segmentation)
     ↓
Clothing Mask
     ↓
K-means Clustering
(Dominant Color Extraction)
     ↓
HSV-Based Color Assignment
     ↓
Predicted Clothing Colors
```

## Key Components

- **SegFormer-B2:** Generates a segmentation mask to isolate the clothing region.
- **K-means Clustering:** Groups clothing pixels to identify dominant colors.
- **HSV-Based Color Assignment:** Maps extracted colors to predefined color categories.

## Supported Colors

The pipeline is designed to identify the following clothing colors:

- White
- Black
- Grey
- Red
- Blue
- Green
- Brown
- Yellow
- Pink
- Purple
- Beige

## Input and Output

**Input:** A cropped torso image extracted from CCTV footage.

**Output:** One or more predicted clothing colors, depending on the colors detected in the clothing region.

## Integration

This pipeline is one of three components in the overall CCTV-based clothing analysis system:

- **Clothing-Type Classification:** Identifies the clothing category.
- **Clothing-Color Analysis:** Identifies the dominant clothing colors.
- **Clothing-Pattern Classification:** Identifies the clothing pattern.

The color predictions can be combined with the outputs of the other two pipelines to generate a unified clothing description.

## Repository Notes

- Ensure that clothing segmentation excludes as much background as possible.
- Keep the predefined color categories consistent throughout the pipeline.
- Evaluate color predictions under different lighting conditions and CCTV image qualities.
- Follow the repository's configuration and scripts for implementation-specific execution instructions.
