# Carwatch

## Overview
This project focuses on detecting vehicles in both video recordings and live camera feeds through motion detection techniques. Utilizing computer vision and OpenCV for advanced video processing and object detection, it can be applied in various scenarios such as monitoring traffic flow, improving security, and analyzing transportation patterns.


## Features and Key Concepts

1. **Preprocessing**
   - Converts frames to grayscale and applies Gaussian blur to reduce noise and detail.
   - **Grayscale Conversion**: Simplifies the image to shades of gray, reducing complexity.
   - **Gaussian Blur**: Smooths the image to minimize noise, making motion patterns more detectable.

2. **Motion Detection**
   - Computes frame differences to highlight changes between consecutive frames.
   - **Difference Calculation**: Detects changes by subtracting the previous frame from the current one.
   - **Thresholding**: Converts the difference image to binary to focus on significant changes.

3. **Contour Enhancement**
   - Applies dilation to enhance contours for better visibility.
   - **Dilation**: Expands white areas in the binary image to make contours more visible and easier to track.

4. **Bounding Box Merging**
   - Merges overlapping or nearby bounding boxes to avoid counting the same vehicle multiple times.
   - **Bounding Boxes**: Draws rectangles around detected contours.
   - **Merging**: Combines overlapping or close boxes to prevent duplicate counts.

5. **Object Tracking**
   - Calculates the distance between bounding boxes to match vehicles across frames.
   - **Euclidean Distance**: Measures the distance between the centers of bounding boxes to track the same vehicle across frames.

6. **Vehicle Counting**
   - Counts and labels detected vehicles based on bounding box size and movement.
   - **Size Threshold**: Determines if a bounding box size indicates a vehicle.
   - **Labeling**: Optionally labels detected vehicles with unique IDs or counts.

7. **Region of Interest (ROI)**
   - Focuses on a specific area of the frame for processing.
   - **Definition**: Concentrates processing on relevant areas to reduce computational load and improve detection accuracy.


## Usage

1. **Setup**: Install required libraries with `pip install opencv-python numpy`.
2. **Run the Script**: Modify the video source in the `input_video` (or `0` for the default camera).
3. **Output**: The processed video is saved as `output_video`. 

A sample video and its processed version are provided in the repository as a proof of concept.

## Disclaimer

The Carwatch project is intended for educational use, and its performance may be affected by factors like video resolution, quality, lighting conditions, and environmental factors. While it offers a basic approach to vehicle detection, it may not achieve full accuracy in every situation. Please use this project with the understanding that results may vary, and no guarantees are provided regarding its effectiveness for particular applications.
