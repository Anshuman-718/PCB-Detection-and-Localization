# AI-Based PCB Detection and Component Localization using YOLOv8

An automated PCB inspection system using **YOLOv8, Python, and OpenCV** to detect an L298N motor driver board, identify its electronic components, and localize pins and terminal connection regions from captured images.

## Project Overview

This project focuses on developing a computer vision-based system for automated visual inspection of an **L298N Motor Driver PCB**.

The system takes PCB images captured using a mobile phone camera as input and processes them through multiple trained deep learning models. These models work together to identify the target PCB, detect individual electronic components, and localize relevant pins and terminals.

Three independently trained detection models are integrated into a single Python application to perform the complete inspection workflow.

## Key Features

- **PCB Detection:** Identifies the L298N motor driver PCB from an input image.
- **Component Detection:** Detects individual electronic components present on the board.
- **Pin and Terminal Localization:** Identifies and localizes relevant pin and terminal connection regions.
- **Multi-Model Integration:** Combines three trained models into a unified detection pipeline.
- **Image-Based Inspection:** Supports processing photographs captured using a mobile phone camera.
- **Embedded Deployment:** Uses Raspberry Pi 4 Model B as the target deployment platform.

## System Workflow

1. Capture an image of the L298N PCB using a mobile phone camera.
2. Transfer the captured image to the computer or Raspberry Pi.
3. Process the input image using the PCB detection model.
4. Identify individual electronic components using the component detection model.
5. Perform pin and terminal localization using the third model.
6. Integrate the outputs from all three detection models.
7. Display and save the processed detection results.

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

### 4. Integrated Application

The three independently trained models are combined into a single Python application. This integration enables the complete PCB inspection workflow through one application.

## Technology Stack

| Technology | Application |
|---|---|
| Python | Main programming language |
| YOLOv8 | Object detection |
| Ultralytics | Model training and inference |
| OpenCV | Image processing and computer vision |
| NumPy | Numerical operations |
| ONNX | Exported model format for deployment |
| Roboflow | Dataset preparation and annotation |
| Raspberry Pi 4 Model B | Target deployment platform |

## Project Structure

```text
PCB-Detection-and-Localization/
│
├── integration/
│   └── main.py
│
├── models/
│   └── raspberry_pi/
│       └── Exported ONNX models
│
├── raspberry_pi/
│   ├── README.md
│   └── Raspberry Pi detection scripts
│
├── results/
│   ├── README.md
│   └── L298N_PCB_Detection_and_Localization_Report.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```

The `integration` directory contains the application that combines the three detection models. The `models` directory stores the exported models for Raspberry Pi deployment. The `raspberry_pi` directory contains deployment scripts and instructions, while `results` contains the final project report.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Anshuman-718/PCB-Detection-and-Localization.git
cd PCB-Detection-and-Localization
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / Raspberry Pi:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Model Files

Place the required trained model files in their respective model directories.

Ensure that the model filenames and paths match those specified in the Python scripts.

### 5. Run the Integrated Application

```bash
python integration/main.py
```

**Note:** The application requires the trained model files and input images at the paths configured in the source code. Dependency versions and model paths may need adjustment depending on the execution platform.

## Raspberry Pi Deployment

The project uses **Raspberry Pi 4 Model B** as its target embedded deployment platform.

The Raspberry Pi deployment uses exported ONNX models and OpenCV for image processing and inference.

The deployment workflow includes:

1. Transferring captured PCB images to the Raspberry Pi.
2. Loading the exported ONNX models.
3. Processing images using the corresponding Python scripts.
4. Generating and saving detection outputs.

For deployment-related scripts and instructions, refer to:

[**Raspberry Pi Deployment Guide**](./raspberry_pi/README.md)

## Sample Results

The project report contains the details of the implementation, model development, system workflow, and experimental results.

The main detection outputs demonstrate:

- L298N motor driver board detection.
- Electronic component identification.
- Pin and terminal localization.
- Integrated multi-model detection.

## Project Report

The detailed project report covers the system architecture, dataset preparation, model development, implementation, Raspberry Pi deployment, and experimental results.

[📄 View or Download Project Report (PDF)](./results/L298N_PCB_Detection_and_Localization_Report.pdf)

## Limitations

- Detection performance depends on image quality, lighting conditions, and PCB orientation.
- The models are specifically developed for the L298N motor driver board and are not intended for general-purpose PCB inspection.
- Detection accuracy may vary with different board layouts, component variations, and image resolutions.
- Raspberry Pi deployment requires consideration of available processing power and memory.

## Future Improvements

- Improve detection performance using a larger and more diverse dataset.
- Enhance pin and terminal localization accuracy.
- Optimize ONNX models for faster inference on Raspberry Pi.
- Develop a graphical user interface for displaying detection results.
- Add automated PCB inspection and fault identification capabilities.
- Explore real-time PCB inspection using a camera module.

## Acknowledgements and Dataset Credits

- **Roboflow:** Used for dataset preparation, image annotation, and model training workflows.
- **L298N Board Detection Dataset:** `visiontest-ryq2d/final-test-qdglq` (Version 2), licensed under CC BY 4.0.
- **Component and Pin Annotation Dataset:** `ui24ec09-anshuman-maurya/component-pin-annotation-sample` (Version 1).

Dataset attribution and licensing terms should be respected when using or redistributing the datasets.

## Project Status

The project includes three independently trained detection models:

1. L298N board detection.
2. Electronic component detection.
3. Pin and terminal localization.

All three models have been integrated into a single Python application. Raspberry Pi 4 Model B is the target deployment platform, with separate deployment scripts and exported ONNX models organized in the repository.

Further improvements in detection accuracy, inference speed, and embedded deployment are possible.

## Author

**Anshuman Maurya**

B.Tech – Electronics and Communication Engineering  
Indian Institute of Information Technology, Surat

- **GitHub:** [Anshuman-718](https://github.com/Anshuman-718)
- **LinkedIn:** [Anshuman Maurya](https://www.linkedin.com/in/anshuman-maurya-b37000324/)

---

If you find this project interesting, feel free to explore the repository and follow its development.
