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


def stable_letter_logic(pred_idx, pred_conf, class_names, state, stable_frames=8, conf_thresh=0.70):
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

    if state["count"] >= stable_frames and state["last_committed"] != current_label:
        state["last_committed"] = current_label
        state["word"] += current_label
        state["count"] = 0
        return current_label

    return None


def draw_text_block(frame, lines, start=(10, 25), line_height=28):
    x, y = start
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


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
        "word": "",
    }

    last_pred = "-"
    last_conf = 0.0
    last_latency_ms = 0.0

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

            if hands is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    roi, box = get_roi_from_hand(frame, hand_landmarks, padding=args.padding)
                    if mp_draw is not None:
                        mp_draw.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(
                    display,
                    "Fallback crop",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 200, 0),
                    2,
                    cv2.LINE_AA,
                )

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

                stable_letter_logic(
                    pred_idx,
                    last_conf,
                    class_names,
                    state,
                    stable_frames=args.stable_frames,
                    conf_thresh=args.conf_thresh,
                )

                if box is not None:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        display,
                        f"{last_pred} ({last_conf:.2f})",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

            lines = [
                f"Model: {model_path.name}",
                f"Prediction: {last_pred}",
                f"Confidence: {last_conf:.2f}",
                f"Latency: {last_latency_ms:.1f} ms",
                f"Word: {state['word']}",
            ]
            draw_text_block(display, lines)

            cv2.imshow("ASL Live UI (TFLite)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                state["word"] = ""
                state["last_committed"] = None
                state["candidate"] = None
                state["count"] = 0
            elif key in (8, 127):  # backspace/delete
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
    parser = argparse.ArgumentParser(description="Run ASL live UI with a TFLite model.")
    parser.add_argument("--model-dir", type=str, default="artifacts", help="Directory containing metadata.json and .tflite model")
    parser.add_argument("--model-name", type=str, default="asl_fp16.tflite", help="TFLite model file name")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device index")
    parser.add_argument("--padding", type=int, default=30, help="Padding around detected hand box")
    parser.add_argument("--stable-frames", type=int, default=8, help="Frames required before committing a letter")
    parser.add_argument("--conf-thresh", type=float, default=0.70, help="Minimum confidence to consider a prediction")
    return parser.parse_args()


if __name__ == "__main__":
    run_live_ui(parse_args())
