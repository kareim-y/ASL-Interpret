import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    TF_IMPORT_ERROR = e
else:
    TF_IMPORT_ERROR = None


DEFAULT_CLASSES = [chr(ord('A') + i) for i in range(26)]

# Colour palette
COL_GREEN   = (0, 220, 100)
COL_YELLOW  = (0, 220, 255)
COL_RED     = (0, 80, 255)
COL_WHITE   = (255, 255, 255)
COL_GREY    = (180, 180, 180)
COL_DARK_BG = (30, 30, 30)
COL_CYAN    = (255, 220, 0)

PANEL_HEIGHT = 150
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX


# Model / inference helpers
def load_metadata(model_dir: Path):
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        class_names = meta.get("class_names", DEFAULT_CLASSES)
        image_size = tuple(meta.get("image_size", [224, 224]))
        return class_names, image_size
    return DEFAULT_CLASSES, (224, 224)


def create_interpreter(model_path: Path):
    if tf is None:
        raise RuntimeError(
            f"TensorFlow import failed: {TF_IMPORT_ERROR}. "
            "For TFLite inference, install tensorflow in this environment."
        )
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess_for_tflite(image_bgr: np.ndarray, image_size):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, image_size)
    image = image_resized.astype(np.float32)
    # NOTE: Do NOT manually scale to [-1,1] here.
    # The MobileNetV2 preprocess_input step is already baked into the
    # TFLite model graph (applied inside the Keras model during training),
    # so the model expects raw [0,255] float32 pixel values as input.
    image = np.expand_dims(image, axis=0)
    return image


def prepare_input_tensor(image_float: np.ndarray, input_details):
    detail = input_details[0]
    dtype = detail["dtype"]

    if np.issubdtype(dtype, np.floating):
        return image_float.astype(dtype)

    # Quantized model
    scale, zero_point = detail["quantization"]
    if scale == 0:
        raise ValueError("Invalid quantization scale 0 in TFLite input tensor.")
    quantized = np.round(image_float / scale + zero_point)
    if dtype == np.uint8:
        quantized = np.clip(quantized, 0, 255)
    elif dtype == np.int8:
        quantized = np.clip(quantized, -128, 127)
    return quantized.astype(dtype)


def decode_output(output_data: np.ndarray, output_detail):
    output = np.squeeze(output_data)
    dtype = output_detail["dtype"]

    if not np.issubdtype(dtype, np.floating):
        scale, zero_point = output_detail["quantization"]
        if scale != 0:
            output = scale * (output.astype(np.float32) - zero_point)
        else:
            output = output.astype(np.float32)
    return output.astype(np.float32)


def get_roi_from_hand(frame_bgr, hand_landmarks, padding=30):
    h, w = frame_bgr.shape[:2]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x1 = max(0, int(min(xs) * w) - padding)
    y1 = max(0, int(min(ys) * h) - padding)
    x2 = min(w, int(max(xs) * w) + padding)
    y2 = min(h, int(max(ys) * h) + padding)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame_bgr[y1:y2, x1:x2]
    box = (x1, y1, x2, y2)
    return crop, box


# Letter commit logic
def stable_letter_logic(pred_idx, pred_conf, class_names, state,
                        stable_frames=8, conf_thresh=0.70, repeat_cooldown=1.5):
    current_label = class_names[pred_idx]

    if pred_conf < conf_thresh:
        state["candidate"] = None
        state["count"] = 0
        return None

    if state["candidate"] == current_label:
        state["count"] += 1
    else:
        state["candidate"] = current_label
        state["count"] = 1

    if state["count"] >= stable_frames:
        now = time.time()
        cooldown_expired = (now - state["last_commit_time"]) >= repeat_cooldown
        is_different = state["last_committed"] != current_label

        if is_different or cooldown_expired:
            state["last_committed"] = current_label
            state["last_commit_time"] = now
            state["word"] += current_label
            state["count"] = 0
            return current_label

    return None


# Drawing helpers
def conf_color(conf, thresh):
    """Return BGR colour that ramps red -> yellow -> green based on confidence."""
    if conf < thresh:
        return COL_RED
    ratio = min((conf - thresh) / (1.0 - thresh + 1e-9), 1.0)
    if ratio < 0.5:
        # red -> yellow
        t = ratio / 0.5
        return (0, int(80 + 140 * t), 255)
    else:
        # yellow -> green
        t = (ratio - 0.5) / 0.5
        return (0, int(220), int(255 * (1 - t)))


def draw_corner_brackets(img, x1, y1, x2, y2, color, thickness=3, length=30):
    """Draw L-shaped corner brackets instead of a full rectangle."""
    ln = length
    # Top-left
    cv2.line(img, (x1, y1), (x1 + ln, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + ln), color, thickness, cv2.LINE_AA)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - ln, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + ln), color, thickness, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + ln, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - ln), color, thickness, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - ln, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - ln), color, thickness, cv2.LINE_AA)

def draw_commit_flash(img, flash_until):
    """Draw a green border flash when a letter is committed."""
    if time.time() < flash_until:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), COL_GREEN, 6, cv2.LINE_AA)


def draw_cooldown_bar(img, state, box, repeat_cooldown, conf_thresh, current_conf):
    """Draw a small progress bar below the detection box showing cooldown."""
    if state["last_committed"] is None or current_conf < conf_thresh:
        return
    elapsed = time.time() - state["last_commit_time"]
    if elapsed >= repeat_cooldown:
        return  # cooldown already done

    ratio = min(elapsed / repeat_cooldown, 1.0)
    x1, y1, x2, y2 = box
    bar_y = y2 + 12
    bar_w = x2 - x1
    bar_h = 8

    # Background
    cv2.rectangle(img, (x1, bar_y), (x2, bar_y + bar_h), (80, 80, 80), -1)
    # Fill
    fill_x = x1 + int(bar_w * ratio)
    cv2.rectangle(img, (x1, bar_y), (fill_x, bar_y + bar_h), COL_CYAN, -1)
    # Border
    cv2.rectangle(img, (x1, bar_y), (x2, bar_y + bar_h), COL_WHITE, 1)


def draw_info_panel(img, model_name, pred, conf, latency_ms):
    """Draw a dark semi-transparent panel at the bottom with stats."""
    h, w = img.shape[:2]
    panel_top = h - PANEL_HEIGHT

    # Semi-transparent dark overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, panel_top), (w, h), COL_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # Thin separator line
    cv2.line(img, (0, panel_top), (w, panel_top), (80, 80, 80), 1)

    # Layout: split into columns
    y_row1 = panel_top + 50
    y_row2 = panel_top + 95
    col1 = 20
    col2 = w // 3
    col3 = (w // 3) * 2

    # Column 1: Model
    cv2.putText(img, "MODEL", (col1, y_row1 - 12), FONT_SM, 1.2, COL_GREY, 1, cv2.LINE_AA)
    cv2.putText(img, model_name, (col1, y_row1 + 20), FONT, 1, COL_WHITE, 1, cv2.LINE_AA)

    # Column 2: Prediction + confidence
    cv2.putText(img, "PREDICTION", (col2, y_row1 - 12), FONT_SM, 1.2, COL_GREY, 1, cv2.LINE_AA)
    pred_color = conf_color(conf, 0.5)
    cv2.putText(img, f"{pred}", (col2, y_row1 + 16), FONT, 1, pred_color, 2, cv2.LINE_AA)
    cv2.putText(img, f"{conf:.0%}", (col2 + 60, y_row1 + 20), FONT, 1, COL_GREY, 1, cv2.LINE_AA)

    # Column 3: Latency
    cv2.putText(img, "LATENCY", (col3, y_row1 - 12), FONT_SM, 1.2, COL_GREY, 1, cv2.LINE_AA)
    cv2.putText(img, f"{latency_ms:.1f} ms", (col3, y_row1 + 20), FONT, 1, COL_WHITE, 1, cv2.LINE_AA)

    # Controls hint
    cv2.putText(img, "Q quit | C clear | SPACE space | BKSP delete",
                (col1, y_row2 + 30), FONT_SM, 0.85, (120, 120, 120), 1, cv2.LINE_AA)


def draw_word_display(img, word):
    """Draw the built-up word as green text in the top-right area."""
    if not word:
        return
    h, w = img.shape[:2]
    label = f"Sentence: {word}"
    text_size = cv2.getTextSize(label, FONT, 1.2, 2)[0]
    tx = w - text_size[0] - 15
    ty = 35
    # Dark background pill for readability
    cv2.rectangle(img, (tx - 10, ty - 25), (tx + text_size[0] + 10, ty + 8),
                  COL_DARK_BG, -1)
    cv2.putText(img, label, (tx, ty), FONT, 1.2, COL_GREEN, 2, cv2.LINE_AA)


# Main loop 
def run_live_ui(args):
    model_dir = Path(args.model_dir)
    model_path = model_dir / args.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    class_names, image_size = load_metadata(model_dir)
    interpreter, input_details, output_details = create_interpreter(model_path)

    hands = None
    mp_hands = None
    mp_draw = None
    if MP_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("Using MediaPipe Hands.")
    else:
        print("MediaPipe Hands API not available. Falling back to center crop.")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    state = {
        "candidate": None,
        "count": 0,
        "last_committed": None,
        "last_commit_time": 0.0,
        "word": "",
    }

    last_pred = "-"
    last_conf = 0.0
    last_latency_ms = 0.0
    flash_until = 0.0  # timestamp until which the green border flash shows

    print("Controls: q=quit, c=clear word, backspace/delete=remove last letter, space=add space")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            roi = None
            box = None
            hand_detected = False

            # Hand detection
            if hands is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    hand_detected = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    roi, box = get_roi_from_hand(frame, hand_landmarks,
                                                 padding=args.padding)
                    if mp_draw is not None:
                        mp_draw.draw_landmarks(display, hand_landmarks,
                                               mp_hands.HAND_CONNECTIONS)

            # Fallback centre crop
            if roi is None or roi.size == 0:
                h, w = frame.shape[:2]
                box_size = int(min(h, w) * 0.6)
                cx, cy = w // 2, h // 2
                x1 = max(cx - box_size // 2, 0)
                y1 = max(cy - box_size // 2, 0)
                x2 = min(cx + box_size // 2, w)
                y2 = min(cy + box_size // 2, h)
                roi = frame[y1:y2, x1:x2]
                box = (x1, y1, x2, y2)
                draw_corner_brackets(display, x1, y1, x2, y2,
                                     COL_YELLOW, thickness=2, length=35)

            # Inference
            committed = None
            if roi is not None and roi.size > 0:
                input_image = preprocess_for_tflite(roi, image_size)
                input_tensor = prepare_input_tensor(input_image, input_details)

                t0 = time.time()
                interpreter.set_tensor(input_details[0]["index"], input_tensor)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]["index"])
                last_latency_ms = (time.time() - t0) * 1000.0

                probs = decode_output(output_data, output_details[0])
                pred_idx = int(np.argmax(probs))
                last_conf = float(probs[pred_idx])
                last_pred = class_names[pred_idx]

                committed = stable_letter_logic(
                    pred_idx,
                    last_conf,
                    class_names,
                    state,
                    stable_frames=args.stable_frames,
                    conf_thresh=args.conf_thresh,
                    repeat_cooldown=args.repeat_cooldown,
                )

                # Trigger flash on commit
                if committed is not None:
                    flash_until = time.time() + 0.3

                # Confidence-coloured corner brackets
                if box is not None:
                    x1, y1, x2, y2 = box
                    color = conf_color(last_conf, args.conf_thresh)
                    draw_corner_brackets(display, x1, y1, x2, y2,
                                         color, thickness=3, length=35)

                    # Prediction label above top-left bracket
                    cv2.putText(
                        display,
                        f"{last_pred} ({last_conf:.0%})",
                        (x1 + 5, max(25, y1 - 12)),
                        FONT, 1.2, color, 2, cv2.LINE_AA,
                    )

            # UI overlays
            draw_commit_flash(display, flash_until)
            draw_word_display(display, state["word"])

            if box is not None:
                draw_cooldown_bar(display, state, box,
                                  args.repeat_cooldown, args.conf_thresh,
                                  last_conf)

            draw_info_panel(display, model_path.name,
                            last_pred, last_conf, last_latency_ms)

            # Show & handle keys
            cv2.imshow("ASL Live UI (TFLite)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                state["word"] = ""
                state["last_committed"] = None
                state["candidate"] = None
                state["count"] = 0
            elif key in (8, 127):  # backspace / delete
                state["word"] = state["word"][:-1]
                state["last_committed"] = None
            elif key == ord(' '):
                state["word"] += " "
                state["last_committed"] = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hands is not None:
            hands.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ASL live UI with a TFLite model.")
    parser.add_argument("--model-dir", type=str, default="artifacts",
                        help="Directory containing metadata.json and .tflite model")
    parser.add_argument("--model-name", type=str, default="asl_fp16.tflite",
                        help="TFLite model file name")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Webcam device index")
    parser.add_argument("--padding", type=int, default=30,
                        help="Padding around detected hand box")
    parser.add_argument("--stable-frames", type=int, default=8,
                        help="Frames required before committing a letter")
    parser.add_argument("--conf-thresh", type=float, default=0.70,
                        help="Minimum confidence to consider a prediction")
    parser.add_argument("--repeat-cooldown", type=float, default=1.5,
                        help="Seconds before the same letter can be committed again")
    return parser.parse_args()


if __name__ == "__main__":
    run_live_ui(parse_args())
