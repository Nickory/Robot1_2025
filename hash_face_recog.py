import gc
import time
from typing import List, Tuple, Optional

import sensor
import image
import lcd
from maix import KPU, GPIO, utils
from fpioa_manager import fm
from board import board_info
from modules import ybserial

# Constants
FACE_PIC_SIZE = 64
THRESHOLD = 80.5
HASH_THRESHOLD = 5
BOUNCE_PROTECTION = 50  # Milliseconds for button debounce
ANCHOR = (0.1075, 0.126875, 0.126875, 0.175, 0.1465625, 0.2246875, 0.1953125,
          0.25375, 0.2440625, 0.351875, 0.341875, 0.4721875, 0.5078125,
          0.6696875, 0.8984375, 1.099687, 2.129062, 2.425937)
DST_POINT = [
    (int(38.2946 * FACE_PIC_SIZE / 112), int(51.6963 * FACE_PIC_SIZE / 112)),
    (int(73.5318 * FACE_PIC_SIZE / 112), int(51.5014 * FACE_PIC_SIZE / 112)),
    (int(56.0252 * FACE_PIC_SIZE / 112), int(71.7366 * FACE_PIC_SIZE / 112)),
    (int(41.5493 * FACE_PIC_SIZE / 112), int(92.3655 * FACE_PIC_SIZE / 112)),
    (int(70.7299 * FACE_PIC_SIZE / 112), int(92.2041 * FACE_PIC_SIZE / 112))
]


def initialize_hardware() -> Tuple[ybserial.YBSerial, KPU, KPU, KPU]:
    """
    Initialize hardware components including LCD, sensor, and KPU models.

    Returns:
        Tuple[ybserial.YBSerial, KPU, KPU, KPU]: Initialized serial, detection KPU,
        landmark KPU, and feature KPU objects.
    """
    # Initialize LCD and sensor
    lcd.init()
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.skip_frames(time=100)

    # Initialize serial communication
    serial = ybserial.YBSerial()

    # Initialize KPU models
    detect_kpu = KPU()
    detect_kpu.load_kmodel("/sd/KPU/yolo_face_detect/face_detect_320x240.kmodel")
    detect_kpu.init_yolo2(
        anchor=ANCHOR, anchor_num=9, img_w=320, img_h=240,
        net_w=320, net_h=240, layer_w=10, layer_h=8,
        threshold=0.7, nms_value=0.2, classes=1
    )

    landmark_kpu = KPU()
    print("Loading landmark model...")
    landmark_kpu.load_kmodel("/sd/KPU/face_recognization/ld5.kmodel")

    feature_kpu = KPU()
    print("Loading feature extraction model...")
    feature_kpu.load_kmodel("/sd/KPU/face_recognization/feature_extraction.kmodel")

    return serial, detect_kpu, landmark_kpu, feature_kpu


def setup_button_interrupt() -> GPIO:
    """
    Configure the boot key interrupt for face registration.

    Returns:
        GPIO: Configured GPIO object for the boot key.
    """
    fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)
    key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)
    key_gpio.irq(
        lambda _: globals().update(start_processing=True) or time.sleep_ms(BOUNCE_PROTECTION),
        GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT
    )
    return key_gpio


def feature_to_hash(feature: List[float]) -> int:
    """
    Convert a feature vector to a 64-bit hash value (integer).

    Args:
        feature (List[float]): The feature vector to convert.

    Returns:
        int: A 64-bit hash value.
    """
    hash_val = 0
    for i, val in enumerate(feature):
        if val >= 0:
            hash_val |= (1 << (i % 64))
    return hash_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Calculate the Hamming distance between two hash values.

    Args:
        hash1 (int): First hash value.
        hash2 (int): Second hash value.

    Returns:
        int: Hamming distance between the two hashes.
    """
    xor_val = hash1 ^ hash2
    distance = 0
    while xor_val:
        distance += xor_val & 1
        xor_val >>= 1
    return distance


def extend_box(x: int, y: int, w: int, h: int, scale: float) -> Tuple[int, int, int, int]:
    """
    Extend the bounding box for face cropping with boundary checks.

    Args:
        x (int): X-coordinate of the top-left corner.
        y (int): Y-coordinate of the top-left corner.
        w (int): Width of the bounding box.
        h (int): Height of the bounding box.
        scale (float): Scaling factor for box extension.

    Returns:
        Tuple[int, int, int, int]: Adjusted (x1, y1, width, height) of the extended box.
    """
    x1_t = x - scale * w
    x2_t = x + w + scale * w
    y1_t = y - scale * h
    y2_t = y + h + scale * h

    x1 = int(x1_t) if x1_t > 1 else 1
    x2 = int(x2_t) if x2_t < 320 else 319
    y1 = int(y1_t) if y1_t > 1 else 1
    y2 = int(y2_t) if y2_t < 240 else 239

    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def process_face(img: image.Image, detect_kpu: KPU, landmark_kpu: KPU,
                 feature_kpu: KPU, record_ftrs: List, hash_record_ftrs: List,
                 feature_img: image.Image) -> Tuple[bool, Optional[str]]:
    """
    Process detected faces, perform recognition, and return recognition status.

    Args:
        img (image.Image): Input image from the sensor.
        detect_kpu (KPU): KPU object for face detection.
        landmark_kpu (KPU): KPU object for landmark detection.
        feature_kpu (KPU): KPU object for feature extraction.
        record_ftrs (List): List of recorded face features.
        hash_record_ftrs (List): List of hashediunea

    Returns:
        Tuple[bool, Optional[str]]: Recognition flag and serial message.
    """
    recog_flag = False
    msg = "N"
    detect_kpu.run_with_output(img)
    detections = detect_kpu.regionlayer_yolo2()

    for detection in detections:
        x, y, w, h = detection[:4]
        x1, y1, cut_img_w, cut_img_h = extend_box(x, y, w, h, scale=0)
        face_cut = img.cut(x1, y1, cut_img_w, cut_img_h)
        face
        cut_128 = face_cut.resize(128, 128)
        face_cut_128.pix_to_ai()

        # Landmark detection
        out = landmark_kpu.run_with_output(face_cut_128, getlist=True)
        face_key_points = [
            (int(KPU.sigmoid(out[2 * j]) * cut_img_w + x1),
             int(KPU.sigmoid(out[2 * j + 1]) * cut_img_h + y1))
            for j in range(5)
        ]

        # Feature extraction
        transform = image.get_affine_transform(face_key_points, DST_POINT)
        image.warp_affine_ai(img, feature_img, transform)
        feature = feature_kpu.run_with_output(feature_img, get_feature=True)

        # Face recognition
        scores = [kpu.feature_compare(record_ftr, feature) for record_ftr in record_ftrs]
        hash_val = feature_to_hash(feature)
        hash_scores = [hamming_distance(hash_val, h) for h in hash_record_ftrs]

        if scores and hash_scores

            max_score = max(scores)
            max_hash_score = min(hash_scores)  # Lower Hamming distance means closer match
            index = scores.index(max_score)
            hash_index = hash_scores.index(max_hash_score)

            if max_score > THRESHOLD and max_hash_score < HASH_THRESHOLD:
                print(f"Face recognized: Index {index}, Score: {max_score:.1f}, Hamming Distance: {max_hash_score}")
                img.draw_string(0, 195, f"Person: {index}, Score: {max_score:.1f}",
                                color=(0, 255, 0), scale=2)
                recog_flag = True
                msg = f"Y{index:02d}"
            else:
                print(f"Unregistered face: Score: {max_score:.1f}, Hamming Distance: {max_hash_score}")
                img.draw_string(0, 195, f"Unregistered, Score: {max_score:.1f}",
                                color=(255, 0, 0), scale=2)
                msg = "N"

        # Register new face if triggered
        if globals().get('start_processing', False):
            record_ftrs.append(feature)
            hash_record_ftrs.append(feature_to_hash(feature))
            print(f"Recorded faces: {len(record_ftrs)}")
            try:
                with open('data.txt', 'w') as file:
                    file.write(str(hash_record_ftrs))
            except Exception as e:
                print(f"Failed to write to data.txt: {e}")
            globals()['start_processing'] = False

        # Draw bounding box
        color = (0, 255, 0) if recog_flag else (255, 255, 255)
        img.draw_rectangle(x, y, w, h, color=color)

        # Clean up
        del face_cut_128
        del face_cut
        del face_key_points
        if scores:
            del scores
        if hash_scores:
            del hash_scores

    return recog_flag, msg


def main() -> None:
    """
    Main loop for face detection and recognition with hash-based comparison.

    Initializes hardware, processes images, and handles face registration and recognition.
    """
    serial, detect_kpu, landmark_kpu, feature_kpu = initialize_hardware()
    setup_button_interrupt()
    clock = time.clock()
    record_ftrs = []
    hash_record_ftrs = []
    feature_img = image.Image(size=(FACE_PIC_SIZE, FACE_PIC_SIZE), copy_to_fb=False)
    feature_img.pix_to_ai()

    try:
        while True:
            gc.collect()
            clock.tick()
            img = sensor.snapshot()
            recog_flag, msg = process_face(img, detect_kpu, landmark_kpu, feature_kpu,
                                           record_ftrs, hash_record_ftrs, feature_img)

            # Send data over serial
            if msg != "N":
                serial.send(f"$08{msg},#")
            else:
                serial.send("#")
            time.sleep_ms(5)

            # Display FPS and instructions
            img.draw_string(0, 0, f"{clock.fps():.1f}fps", color=(0, 60, 255), scale=2.0)
            img.draw_string(0, 215, "Press boot key to register face",
                            color=(255, 100, 0), scale=2.0)
            lcd.display(img)

    finally:
        # Clean up KPU resources
        detect_kpu.deinit()
        landmark_kpu.deinit()
        feature_kpu.deinit()


if __name__ == "__main__":
    main()