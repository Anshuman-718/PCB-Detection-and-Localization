# Raspberry Pi Deployment

## Overview
The project is designed to run on a Raspberry Pi 4 Model B for automated inspection of an L298N motor driver PCB.

The Raspberry Pi deployment uses exported ONNX models and Python scripts with OpenCV for image processing and detection.

## Hardware Requirements
- Raspberry Pi 4 Model B
- Raspberry Pi power supply
- microSD card
- Camera or mobile phone for capturing PCB images
- Monitor, keyboard and mouse (optional)

## Software Requirements
- Raspberry Pi OS
- Python 3
- OpenCV
- NumPy
- ONNX model files

## Deployment Workflow
1. Capture an image of the L298N PCB using a mobile phone or camera.
2. Transfer the image to the Raspberry Pi.
3. Load the image using the Python detection script.
4. Detect the L298N board.
5. Identify electronic components on the board.
6. Localize the pins and terminal connections.
7. Save and inspect the generated detection results.

## Model Files
Place the exported ONNX models in the `models/raspberry_pi/` directory.

The model files should correspond to the board detection, component detection and pin-terminal localization modules.

## Running the Project
1. Clone or download this repository.
2. Install the required Python dependencies.
3. Copy the exported ONNX models into the expected model directory.
4. Place test images in the input image directory.
5. Run the corresponding Raspberry Pi Python script.

**Note:** Update model and image paths in the scripts according to the directory structure on your Raspberry Pi.

## Target Platform
Raspberry Pi 4 Model B

This deployment aims to demonstrate local PCB detection and component localization without requiring a cloud-based processing service.
