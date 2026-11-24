import os
import cv2
import torch
from collections import defaultdict
from ultralytics import YOLO


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = r"C:\Users\DELL\PycharmProjects\PythonProject\PCB_DETECTION(L298N)"

MODEL_DIR = os.path.join(BASE_DIR, "models")

BOARD_MODEL_PATH = os.path.join(MODEL_DIR, "board_detector.pt")
COMPONENT_MODEL_PATH = os.path.join(MODEL_DIR, "component_detector.pt")
PIN_MODEL_PATH = os.path.join(MODEL_DIR, "pin_localization.pt")

INPUT_FOLDER = os.path.join(BASE_DIR, "test_images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_images")


OUTPUT_FOLDERS = {
    "board": os.path.join(OUTPUT_FOLDER, "board_only"),
    "component": os.path.join(OUTPUT_FOLDER, "component_only"),
    "pin": os.path.join(OUTPUT_FOLDER, "pin_only"),
    "combined": os.path.join(OUTPUT_FOLDER, "combined"),
    "crop": os.path.join(OUTPUT_FOLDER, "board_crops")
}

for folder in OUTPUT_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)


# ============================================================
# 2. DETECTION SETTINGS
# ============================================================

BOARD_CONF = 0.25
COMPONENT_CONF = 0.25
PIN_CONF = 0.20

BOARD_IMGSZ = 640
COMPONENT_IMGSZ = 640
PIN_IMGSZ = 800

CROP_PADDING = 10

IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
)


# ============================================================
# 3. EXPECTED COMPONENT COUNTS
# ============================================================

# Enter actual expected quantities for your L298N board.
# Use exact class names printed by the component model.
#
# Example:
# "driver_ic": 1
#
# None means expected count has not been configured.

EXPECTED_COMPONENT_COUNTS = {
    "capacitor": 2,
    "diode": 8,
    "driver_ic": 1,
    "header_pin": 2,
    "heatsink": 1,
    "jumper": 2,
    "power_led": 1,
    "resistor": 1,
    "terminal_block": 3,
    "voltage_regulator": 1
}

# ============================================================
# 4. SELECT DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("\nUsing device:", "GPU" if DEVICE == 0 else "CPU")


# ============================================================
# 5. LOAD MODELS
# ============================================================

print("\nLoading trained models...")

for path in [
    BOARD_MODEL_PATH,
    COMPONENT_MODEL_PATH,
    PIN_MODEL_PATH
]:

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")


board_model = YOLO(BOARD_MODEL_PATH)
component_model = YOLO(COMPONENT_MODEL_PATH)
pin_model = YOLO(PIN_MODEL_PATH)

print("All three models loaded successfully!")


# ============================================================
# 6. DISPLAY MODEL CLASSES
# ============================================================

def display_classes(model, model_name):

    print(f"\n{model_name} classes:")

    for class_id, class_name in model.names.items():
        print(f"{class_id}: {class_name}")


display_classes(board_model, "BOARD MODEL")
display_classes(component_model, "COMPONENT MODEL")
display_classes(pin_model, "PIN MODEL")


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def get_class_name(model, class_id):

    return str(model.names[int(class_id)])


def clip_box(box, width, height):

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))

    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return x1, y1, x2, y2


def draw_detection(image, box, label, color, thickness=2):

    height, width = image.shape[:2]

    x1, y1, x2, y2 = clip_box(box, width, height)

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    if label:

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        text_thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness
        )

        label_y = max(y1, text_height + 5)

        cv2.rectangle(
            image,
            (x1, label_y - text_height - 5),
            (x1 + text_width + 5, label_y + baseline),
            color,
            -1
        )

        cv2.putText(
            image,
            label,
            (x1 + 2, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )


def update_class_statistics(statistics, model, boxes):

    if boxes is None:
        return

    for box in boxes:

        class_id = int(box.cls[0].item())

        class_name = get_class_name(model, class_id)

        confidence = float(box.conf[0].item())

        statistics[class_name]["count"] += 1

        statistics[class_name]["confidence_sum"] += confidence


def print_class_statistics(statistics, model_name):

    print(f"\n{model_name} CLASS-WISE RESULTS")

    print("-" * 60)

    if not statistics:
        print("No detections.")
        return

    for class_name, values in sorted(statistics.items()):

        count = values["count"]

        average_confidence = (
            values["confidence_sum"] / count
        )

        print(
            f"{class_name:<25} "
            f"Count: {count:<5} "
            f"Avg confidence: {average_confidence:.3f}"
        )


def save_image(image, folder, filename):

    output_path = os.path.join(folder, filename)

    success = cv2.imwrite(output_path, image)

    if success:
        print("Saved:", output_path)

    else:
        print("Could not save:", output_path)


# ============================================================
# 8. COMPONENT COUNT VERIFICATION
# ============================================================

def verify_component_counts(statistics, model):

    print("\n")
    print("=" * 85)
    print("COMPONENT COUNT VERIFICATION")
    print("=" * 85)

    print(
        f"{'Component':<25}"
        f"{'Expected':<12}"
        f"{'Detected':<12}"
        f"{'Difference':<12}"
        f"{'Status'}"
    )

    print("-" * 85)

    total_expected = 0
    total_detected = 0
    configured_classes = 0

    missing_classes = []
    extra_classes = []

    all_configured_match = True

    for class_id, class_name in model.names.items():

        detected = statistics[class_name]["count"]

        expected = EXPECTED_COMPONENT_COUNTS.get(class_name)

        if expected is None:

            print(
                f"{class_name:<25}"
                f"{'Not set':<12}"
                f"{detected:<12}"
                f"{'-':<12}"
                f"CONFIGURE"
            )

            continue

        difference = expected - detected

        configured_classes += 1

        total_expected += expected
        total_detected += detected

        if difference == 0:

            status = "PASS"

        elif difference > 0:

            status = f"MISSING {difference}"

            missing_classes.append(
                (class_name, difference)
            )

            all_configured_match = False

        else:

            status = f"EXTRA {abs(difference)}"

            extra_classes.append(
                (class_name, abs(difference))
            )

            all_configured_match = False

        print(
            f"{class_name:<25}"
            f"{expected:<12}"
            f"{detected:<12}"
            f"{difference:<12}"
            f"{status}"
        )

    print("-" * 85)

    print("Total expected:", total_expected)
    print("Total detected:", total_detected)

    if configured_classes == 0:

        print("\nNo expected component counts configured.")

    else:

        print("\nVERIFICATION SUMMARY")

        if all_configured_match:

            print("Overall count: PASS")

        else:

            print("Overall count: MISMATCH")

        if missing_classes:

            print("\nMissing component detections:")

            for name, count in missing_classes:

                print(f"{name}: {count} missing")

        if extra_classes:

            print("\nExtra component detections:")

            for name, count in extra_classes:

                print(f"{name}: {count} extra")

    print("=" * 85)


# ============================================================
# 9. PROCESS ONE IMAGE
# ============================================================

def process_image(image_path):

    filename = os.path.basename(image_path)

    print("\n" + "=" * 75)
    print("Processing:", filename)
    print("=" * 75)

    original_image = cv2.imread(image_path)

    if original_image is None:

        print("Unable to read image. Skipping...")
        return False

    height, width = original_image.shape[:2]

    # Separate visualization canvases

    board_output = original_image.copy()
    component_output = original_image.copy()
    pin_output = original_image.copy()
    combined_output = original_image.copy()

    # Statistics

    component_statistics = defaultdict(
        lambda: {"count": 0, "confidence_sum": 0.0}
    )

    pin_statistics = defaultdict(
        lambda: {"count": 0, "confidence_sum": 0.0}
    )

    # --------------------------------------------------------
    # STEP 1: BOARD DETECTION
    # --------------------------------------------------------

    print("\nSTEP 1: Running board detector...")

    board_results = board_model.predict(
        source=original_image,
        conf=BOARD_CONF,
        imgsz=BOARD_IMGSZ,
        device=DEVICE,
        verbose=False
    )

    board_boxes = board_results[0].boxes

    if board_boxes is None or len(board_boxes) == 0:

        print("No board detected.")

        save_image(
            original_image,
            OUTPUT_FOLDERS["combined"],
            filename
        )

        return False

    # Find L298N board

    l298n_candidates = []

    for box in board_boxes:

        class_id = int(box.cls[0].item())

        class_name = get_class_name(
            board_model,
            class_id
        )

        confidence = float(box.conf[0].item())

        if "l298" in class_name.lower().replace(" ", ""):

            l298n_candidates.append(
                (box, confidence, class_name)
            )

    if not l298n_candidates:

        print("No class named L298N found.")

        return False

    # Select highest-confidence L298N detection

    selected_box, board_confidence, board_class = max(
        l298n_candidates,
        key=lambda item: item[1]
    )

    coordinates = selected_box.xyxy[0].cpu().numpy()

    bx1, by1, bx2, by2 = clip_box(
        coordinates,
        width,
        height
    )

    if bx2 <= bx1 or by2 <= by1:

        print("Invalid board bounding box.")
        return False

    print("Board detected:", board_class)

    print(
        "Board confidence:",
        round(board_confidence, 3)
    )

    board_label = f"L298N {board_confidence:.2f}"

    # Board-only output

    draw_detection(
        board_output,
        (bx1, by1, bx2, by2),
        board_label,
        (255, 0, 0),
        3
    )

    # Combined output

    draw_detection(
        combined_output,
        (bx1, by1, bx2, by2),
        board_label,
        (255, 0, 0),
        3
    )

    # --------------------------------------------------------
    # STEP 2: BOARD CROPPING
    # --------------------------------------------------------

    print("\nSTEP 2: Cropping L298N board...")

    crop_x1 = max(0, bx1 - CROP_PADDING)
    crop_y1 = max(0, by1 - CROP_PADDING)

    crop_x2 = min(width, bx2 + CROP_PADDING)
    crop_y2 = min(height, by2 + CROP_PADDING)

    board_crop = original_image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    if board_crop.size == 0:

        print("Board crop is empty.")
        return False

    print("Board crop size:", board_crop.shape)

    save_image(
        board_crop,
        OUTPUT_FOLDERS["crop"],
        filename
    )

    # --------------------------------------------------------
    # STEP 3: COMPONENT DETECTION
    # --------------------------------------------------------

    print("\nSTEP 3: Running component detector...")

    component_results = component_model.predict(
        source=board_crop,
        conf=COMPONENT_CONF,
        imgsz=COMPONENT_IMGSZ,
        device=DEVICE,
        verbose=False
    )

    component_boxes = component_results[0].boxes

    update_class_statistics(
        component_statistics,
        component_model,
        component_boxes
    )

    component_count = 0

    if component_boxes is not None:

        for box in component_boxes:

            coordinates = box.xyxy[0].cpu().numpy()

            # Convert crop coordinates to original coordinates

            x1, y1, x2, y2 = coordinates

            x1 += crop_x1
            x2 += crop_x1

            y1 += crop_y1
            y2 += crop_y1

            class_id = int(box.cls[0].item())

            class_name = get_class_name(
                component_model,
                class_id
            )

            confidence = float(box.conf[0].item())

            label = f"{class_name} {confidence:.2f}"

            # Component-only output

            draw_detection(
                component_output,
                (x1, y1, x2, y2),
                label,
                (0, 255, 0),
                2
            )

            # Combined output

            draw_detection(
                combined_output,
                (x1, y1, x2, y2),
                label,
                (0, 255, 0),
                2
            )

            component_count += 1

    print("Total components detected:", component_count)

    # --------------------------------------------------------
    # STEP 4: PIN AND TERMINAL LOCALIZATION
    # --------------------------------------------------------

    print("\nSTEP 4: Running pin localization model...")

    pin_results = pin_model.predict(
        source=board_crop,
        conf=PIN_CONF,
        imgsz=PIN_IMGSZ,
        device=DEVICE,
        verbose=False
    )

    pin_boxes = pin_results[0].boxes

    update_class_statistics(
        pin_statistics,
        pin_model,
        pin_boxes
    )

    pin_count = 0

    if pin_boxes is not None:

        for box in pin_boxes:

            coordinates = box.xyxy[0].cpu().numpy()

            # Convert crop coordinates to original coordinates

            x1, y1, x2, y2 = coordinates

            x1 += crop_x1
            x2 += crop_x1

            y1 += crop_y1
            y2 += crop_y1

            class_id = int(box.cls[0].item())

            class_name = get_class_name(
                pin_model,
                class_id
            )

            confidence = float(box.conf[0].item())

            label = f"{class_name} {confidence:.2f}"

            # Pin-only output

            draw_detection(
                pin_output,
                (x1, y1, x2, y2),
                label,
                (0, 165, 255),
                1
            )

            # Combined output

            draw_detection(
                combined_output,
                (x1, y1, x2, y2),
                "",
                (0, 165, 255),
                1
            )

            pin_count += 1

    print("Total pins/contacts detected:", pin_count)

    # --------------------------------------------------------
    # STEP 5: CLASS-WISE STATISTICS
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print_class_statistics(
        component_statistics,
        "COMPONENT DETECTOR"
    )

    print_class_statistics(
        pin_statistics,
        "PIN LOCALIZATION MODEL"
    )

    # --------------------------------------------------------
    # STEP 6: COMPONENT COUNT VERIFICATION
    # --------------------------------------------------------

    verify_component_counts(
        component_statistics,
        component_model
    )

    # --------------------------------------------------------
    # STEP 7: TOTAL DETECTION SUMMARY
    # --------------------------------------------------------

    print("\nTOTAL DETECTIONS")
    print("-" * 60)

    print("Board:", 1)
    print("Components:", component_count)
    print("Pins/contacts:", pin_count)

    # --------------------------------------------------------
    # STEP 8: SAVE ALL FOUR OUTPUTS
    # --------------------------------------------------------

    print("\nSTEP 8: Saving output images...")

    save_image(
        board_output,
        OUTPUT_FOLDERS["board"],
        filename
    )

    save_image(
        component_output,
        OUTPUT_FOLDERS["component"],
        filename
    )

    save_image(
        pin_output,
        OUTPUT_FOLDERS["pin"],
        filename
    )

    save_image(
        combined_output,
        OUTPUT_FOLDERS["combined"],
        filename
    )

    print("\nImage processing completed:", filename)

    return True


# ============================================================
# 10. PROCESS ALL TEST IMAGES
# ============================================================

def main():

    if not os.path.isdir(INPUT_FOLDER):

        raise FileNotFoundError(
            f"Input folder not found: {INPUT_FOLDER}"
        )

    image_files = sorted([

        filename

        for filename in os.listdir(INPUT_FOLDER)

        if filename.lower().endswith(IMAGE_EXTENSIONS)

    ])

    if not image_files:

        print("No images found in test_images folder.")
        return

    print("\nTotal images found:", len(image_files))

    successful_images = 0
    failed_images = 0

    for filename in image_files:

        image_path = os.path.join(
            INPUT_FOLDER,
            filename
        )

        try:

            success = process_image(image_path)

            if success:
                successful_images += 1

            else:
                failed_images += 1

        except Exception as error:

            print("\nERROR processing:", filename)

            print(type(error).__name__, error)

            failed_images += 1

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("ALL IMAGES PROCESSED")
    print("=" * 75)

    print("Total images:", len(image_files))
    print("Successful images:", successful_images)
    print("Failed images:", failed_images)

    print("\nOUTPUT FOLDERS")

    for name, folder in OUTPUT_FOLDERS.items():

        print(f"{name}: {folder}")

    print("\nProcessing completed.")


# ============================================================
# 11. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
