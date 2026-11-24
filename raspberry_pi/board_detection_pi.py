
import cv2
import numpy as np
import time
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path.home() / "L298N_Project"

MODEL_PATH = BASE_DIR / "models" / "board_detector.onnx"
INPUT_DIR = BASE_DIR / "test_images"

OUTPUT_DIR = BASE_DIR / "output_images"
BOARD_OUTPUT_DIR = OUTPUT_DIR / "board_only"

CROP_OUTPUT_DIR = OUTPUT_DIR / "board_crops"

BOARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DETECTION SETTINGS
# ============================================================

BOARD_CONF = 0.25
BOARD_IMGSZ = 640

NMS_THRESHOLD = 0.70
CROP_PADDING = 10

CLASS_NAMES = [
    "Arduino uno",
    "Arduino-Mega",
    "L298N"
]

TARGET_CLASS = "L298N"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\nLoading L298N board detector...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))

print("Board model loaded successfully!")


# ============================================================
# 4. LETTERBOX PREPROCESSING
# ============================================================

def letterbox(image, new_size=640):

    height, width = image.shape[:2]

    scale = min(
        new_size / height,
        new_size / width
    )

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    pad_width = new_size - new_width
    pad_height = new_size - new_height

    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))

    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))

    output = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return output, scale, left, top


# ============================================================
# 5. YOLO OUTPUT DECODING
# ============================================================

def decode_output(output, scale, pad_x, pad_y, original_shape):

    height, width = original_shape[:2]

    predictions = output[0].T

    boxes = []
    confidences = []
    class_ids = []

    for prediction in predictions:

        scores = prediction[4:]

        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence < BOARD_CONF:
            continue

        if class_id >= len(CLASS_NAMES):
            continue

        # YOLOv8 output: center_x, center_y, width, height

        center_x, center_y, box_width, box_height = (
            prediction[:4]
        )

        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2

        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2

        # Remove letterbox padding and restore original scale

        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale

        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))

        x2 = max(0, min(int(round(x2)), width))
        y2 = max(0, min(int(round(y2)), height))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        confidences.append(confidence)
        class_ids.append(class_id)

    return boxes, confidences, class_ids


# ============================================================
# 6. DRAW DETECTION
# ============================================================

def draw_detection(image, box, label):

    x1, y1, x2, y2 = box

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3
    )

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


# ============================================================
# 7. PROCESS ONE IMAGE
# ============================================================

def process_image(image_path):

    filename = image_path.name

    print("\n" + "=" * 60)
    print("Processing:", filename)
    print("=" * 60)

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        return False

    original = image.copy()

    print("Original image size:", image.shape)

    # --------------------------------------------------------
    # STEP 1: PREPROCESSING
    # --------------------------------------------------------

    processed_image, scale, pad_x, pad_y = letterbox(
        image,
        BOARD_IMGSZ
    )

    blob = cv2.dnn.blobFromImage(
        processed_image,
        scalefactor=1 / 255.0,
        size=(BOARD_IMGSZ, BOARD_IMGSZ),
        swapRB=True,
        crop=False
    )

    # --------------------------------------------------------
    # STEP 2: MODEL INFERENCE
    # --------------------------------------------------------

    net.setInput(blob)

    start_time = time.perf_counter()

    output = net.forward()

    inference_time = (
        time.perf_counter() - start_time
    ) * 1000

    print("Raw output shape:", output.shape)

    print(
        f"Inference time: {inference_time:.2f} ms"
    )

    # --------------------------------------------------------
    # STEP 3: DECODE PREDICTIONS
    # --------------------------------------------------------

    boxes, confidences, class_ids = decode_output(
        output,
        scale,
        pad_x,
        pad_y,
        original.shape
    )

    print("Candidates above confidence threshold:", len(boxes))

    # --------------------------------------------------------
    # STEP 4: SELECT L298N DETECTIONS
    # --------------------------------------------------------

    target_boxes = []
    target_confidences = []

    for box, confidence, class_id in zip(
        boxes,
        confidences,
        class_ids
    ):

        class_name = CLASS_NAMES[class_id]

        if class_name.lower().replace(" ", "") == "l298n":

            target_boxes.append(box)
            target_confidences.append(confidence)

    if not target_boxes:

        print("No L298N board detected.")

        return False

    # --------------------------------------------------------
    # STEP 5: NON-MAXIMUM SUPPRESSION
    # --------------------------------------------------------

    indices = cv2.dnn.NMSBoxes(
        target_boxes,
        target_confidences,
        BOARD_CONF,
        NMS_THRESHOLD
    )

    if len(indices) == 0:

        print("No valid detection after NMS.")
        return False

    indices = np.array(indices).flatten()

    # Select highest-confidence remaining detection

    best_index = max(
        indices,
        key=lambda index: target_confidences[index]
    )

    selected_box = target_boxes[best_index]

    confidence = target_confidences[best_index]

    x, y, w, h = selected_box

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    print("\nL298N board detected!")
    print("Confidence:", round(confidence, 4))
    print("Bounding box:", (x1, y1, x2, y2))

    # --------------------------------------------------------
    # STEP 6: DRAW BOARD DETECTION
    # --------------------------------------------------------

    label = f"L298N {confidence:.2f}"

    draw_detection(
        image,
        (x1, y1, x2, y2),
        label
    )

    # --------------------------------------------------------
    # STEP 7: CROP BOARD
    # --------------------------------------------------------

    height, width = original.shape[:2]

    crop_x1 = max(0, x1 - CROP_PADDING)
    crop_y1 = max(0, y1 - CROP_PADDING)

    crop_x2 = min(width, x2 + CROP_PADDING)
    crop_y2 = min(height, y2 + CROP_PADDING)

    board_crop = original[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    if board_crop.size == 0:

        print("Invalid board crop.")
        return False

    print("Board crop size:", board_crop.shape)

    # --------------------------------------------------------
    # STEP 8: SAVE OUTPUT
    # --------------------------------------------------------

    output_path = BOARD_OUTPUT_DIR / filename
    crop_path = CROP_OUTPUT_DIR / filename

    cv2.imwrite(str(output_path), image)
    cv2.imwrite(str(crop_path), board_crop)

    print("\nSaved board detection:", output_path)
    print("Saved board crop:", crop_path)

    print("\nBoard detection completed successfully!")

    return True


# ============================================================
# 8. MAIN
# ============================================================

def main():

    images = sorted([
        path
        for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        print("No test images found.")
        return

    print("\nTotal images available:", len(images))

    # Test only the first image initially

    success = process_image(images[0])

    print("\n" + "=" * 60)

    if success:
        print("BOARD DETECTION TEST PASSED")
    else:
        print("BOARD DETECTION TEST FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()
