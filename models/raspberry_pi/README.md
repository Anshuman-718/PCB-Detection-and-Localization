# Raspberry Pi ONNX Models

This directory contains the exported ONNX models used by the Raspberry Pi deployment scripts.

## Required Models

1. `board_detector.onnx` – Detects the L298N motor driver board.
2. `component_detector.onnx` – Detects electronic components on the L298N board.
3. `pin_localization.onnx` – Localizes pins and terminal connections.

Place the corresponding ONNX files in this directory.

**Note:** These are the expected model names. If your exported files have different names, keep their actual names and update the corresponding paths in the Python scripts.
