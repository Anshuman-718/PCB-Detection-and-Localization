
import cv2
import numpy as np
import time
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path.home() / "L298N_Project"

MODEL_PATH = PROJECT_DIR / "models" / "component_detector.onnx"

CROP_DIR = PROJECT_DIR / "output_images" / "board_crops"

OUTPUT_DIR = PROJECT_DIR / "output_images" / "component_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL SETTINGS
# ============================================================

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.70
IMG_SIZE = 640

CLASS_NAMES = [
    "capacitor",
    "diode",
    "driver_ic",
    "header_pin",
    "heatsink",
    "jumper",
    "power_led",
    "resistor",
    "terminal_block",
    "voltage_regulator"
]

# ============================================================
# LETTERBOX PREPROCESSING
# ============================================================

def letterbox(image, new_size=640):

    height, width = image.shape[:2]

    scale = min(new_size / height, new_size / width)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    pad_x = (new_size - new_width) / 2
    pad_y = (new_size - new_height) / 2

    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))

    processed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return processed, scale, left, top


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading component detector...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))

print("Component model loaded successfully!")

# ============================================================
# FIND BOARD CROP
# ============================================================

crop_images = sorted(CROP_DIR.glob("*.jpg"))

if not crop_images:
    raise FileNotFoundError(
        f"No board crop images found in: {CROP_DIR}"
    )

# Test the first board crop
image_path = crop_images[0]

print("\n" + "=" * 60)
print(f"Processing: {image_path.name}")
print("=" * 60)

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Unable to read board crop image.")

original = image.copy()

height, width = image.shape[:2]

print(f"Board crop size: {image.shape}")

# ============================================================
# PREPROCESS IMAGE
# ============================================================

processed, scale, pad_x, pad_y = letterbox(
    image,
    IMG_SIZE
)

blob = cv2.dnn.blobFromImage(
    processed,
    scalefactor=1 / 255.0,
    size=(IMG_SIZE, IMG_SIZE),
    swapRB=True,
    crop=False
)

# ============================================================
# RUN INFERENCE
# ============================================================

net.setInput(blob)

start_time = time.time()

output = net.forward()

inference_time = (time.time() - start_time) * 1000

print(f"Raw output shape: {output.shape}")
print(f"Inference time: {inference_time:.2f} ms")

# ============================================================
# DECODE YOLO OUTPUT
# ============================================================

predictions = output[0].T

boxes = []
confidences = []
class_ids = []

for prediction in predictions:

    scores = prediction[4:]

    class_id = int(np.argmax(scores))
    confidence = float(scores[class_id])

    if confidence < CONF_THRESHOLD:
        continue

    center_x, center_y, box_width, box_height = prediction[:4]

    x1 = center_x - box_width / 2
    y1 = center_y - box_height / 2

    x2 = center_x + box_width / 2
    y2 = center_y + box_height / 2

    # Remove letterbox padding
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    # Clip coordinates to original crop dimensions
    x1 = int(np.clip(round(x1), 0, width - 1))
    y1 = int(np.clip(round(y1), 0, height - 1))
    x2 = int(np.clip(round(x2), 0, width - 1))
    y2 = int(np.clip(round(y2), 0, height - 1))

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        continue

    boxes.append([x1, y1, box_w, box_h])
    confidences.append(confidence)
    class_ids.append(class_id)

print(
    f"Candidates above confidence threshold: {len(boxes)}"
)

# ============================================================
# CLASS-WISE NON-MAXIMUM SUPPRESSION
# ============================================================

final_detections = []

for class_id in range(len(CLASS_NAMES)):

    indices = [
        i for i, detected_class in enumerate(class_ids)
        if detected_class == class_id
    ]

    if not indices:
        continue

    class_boxes = [boxes[i] for i in indices]
    class_confidences = [confidences[i] for i in indices]

    selected = cv2.dnn.NMSBoxes(
        class_boxes,
        class_confidences,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(selected) > 0:

        for selected_index in np.array(selected).flatten():

            original_index = indices[int(selected_index)]

            final_detections.append({
                "class_id": class_ids[original_index],
                "confidence": confidences[original_index],
                "box": boxes[original_index]
            })

# ============================================================
# DRAW DETECTIONS
# ============================================================

result = original.copy()

class_counts = {name: 0 for name in CLASS_NAMES}

for detection in final_detections:

    class_id = detection["class_id"]
    confidence = detection["confidence"]

    x, y, box_w, box_h = detection["box"]

    class_name = CLASS_NAMES[class_id]

    class_counts[class_name] += 1

    label = f"{class_name}: {confidence:.2f}"

    cv2.rectangle(
        result,
        (x, y),
        (x + box_w, y + box_h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        label,
        (x, max(y - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

# ============================================================
# PRINT DETECTION STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("COMPONENT DETECTION RESULTS")
print("=" * 60)

print(f"Total components detected: {len(final_detections)}")

for class_name, count in class_counts.items():

    print(f"{class_name:22s}: {count}")

# ============================================================
# SAVE OUTPUT
# ============================================================

output_path = OUTPUT_DIR / image_path.name

success = cv2.imwrite(str(output_path), result)

if not success:
    raise RuntimeError("Unable to save component detection image.")

print("\nSaved component detection image:")
print(output_path)

print("\n" + "=" * 60)
print("COMPONENT DETECTION TEST COMPLETED")
print("=" * 60)
