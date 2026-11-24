**AI-Based PCB Detection and Component Localization using YOLOv8**

An automated PCB inspection system using **YOLOv8, Python, and OpenCV** to detect an L298N motor driver board, identify its electronic components, and localize pins and terminal connection regions from captured images.

## Project Overview

This project focuses on developing a computer vision-based system for automated visual inspection of an **L298N Motor Driver PCB**.

The system takes PCB images captured using a mobile phone camera as input and processes them through multiple trained deep learning models. These models work together to identify the target PCB, detect individual electronic components, and localize relevant pins and terminals.

The project integrates three trained detection models into a single Python application.

## Key Features

- **PCB Detection:** Identifies the L298N motor driver PCB from an input image.
- **Component Detection:** Detects individual electronic components present on the board.
- **Pin and Terminal Localization:** Identifies and localizes relevant pin and terminal connection regions.
- **Multi-Model Integration:** Combines three trained models into a unified detection pipeline.
- **Image-Based Inspection:** Supports processing photographs captured using a mobile phone camera.
- **Embedded Deployment Target:** Developed with Raspberry Pi 4 Model B as the intended deployment platform.

## System Workflow

1. Capture an image of the L298N PCB using a mobile phone camera.
2. Transfer the captured image to the computer running the application.
3. Process the input image using the trained PCB detection model.
4. Identify individual electronic components using the component detection model.
5. Perform pin and terminal localization using the third model.
6. Integrate and display the detection results through the Python application.

## Detection Modules

### 1. PCB Detection

The first trained YOLOv8 model identifies the target L298N motor driver board from the input image.

### 2. Electronic Component Detection

The second model identifies individual components on the L298N PCB, including:

- Driver IC
- Heat sink
- Capacitors
- Diodes
- Terminal blocks
- Jumper pins
- Header pins
- Power LED
- Resistors
- Voltage regulator IC

### 3. Pin and Terminal Localization

The third model focuses on identifying and localizing relevant pin and terminal connection regions on the PCB.

## Technology Stack

| Technology | Application |
|---|---|
| Python | Main programming language |
| YOLOv8 | Object detection |
| Ultralytics | Model training and inference |
| OpenCV | Image processing |
| NumPy | Numerical operations |
| Raspberry Pi 4 Model B | Target deployment platform |

## Project Structure

```text
PCB-Detection-and-Localization/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── board_detection.pt
│   ├── component_detection.pt
│   └── pin_terminal_detection.pt
│
├── integration/
│   └── main.py
│
├── utils/
│   └── ...
│
├── sample_images/
│   └── ...
│
└── results/
    └── ...
```

*Note: The structure above represents the planned organization. Actual filenames and folders may differ.*

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- Git
- Required Python libraries
- Trained YOLOv8 model weights

### 1. Clone the repository

```bash
git clone https://github.com/Anshuman-718/PCB-Detection-and-Localization.git
```

### 2. Navigate to the project directory

```bash
cd PCB-Detection-and-Localization
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

After placing the trained model weights in their configured locations, execute the main application:

```bash
python integration/main.py
```

*The execution command and model paths should be adjusted according to the final repository structure.*

## Sample Results

This section will contain actual project outputs demonstrating:

- Original mobile-captured PCB images.
- L298N board detection results.
- Individual component detection results.
- Pin and terminal localization results.
- Integrated system output.

## Target Deployment

The project is being developed with **Raspberry Pi 4 Model B** as its target deployment platform, with the objective of enabling image-based PCB inspection on an embedded computing system.

## Future Scope

- Improve detection performance under different lighting conditions and camera angles.
- Enhance pin and terminal localization accuracy.
- Optimize inference for Raspberry Pi deployment.
- Develop a standalone PCB inspection interface.
- Extend the system to support additional PCB types.

## Project Status

**Ongoing Development**

The PCB detection, component detection, and multi-model integration stages are being developed as part of the project. Further improvements and deployment testing are planned.

## Project Report

The detailed project report covers the system architecture, dataset preparation, model development, implementation, Raspberry Pi deployment, and experimental results.

[📄 View or Download Project Report (PDF)](./results/L298N_PCB_Detection_and_Localization_Report.pdf))**

If you find this project interesting, feel free to explore the repository and follow its development.
