import cv2
import os

save_dir = "Data200"
cap = cv2.VideoCapture(0)
current_letter = "A"
count = 0

print("Press a letter key to set which letter you're signing.")
print("Press SPACE to capture. Press ESC to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    box_size = int(min(h, w) * 0.6)
    cx, cy = w // 2, h // 2
    x1 = cx - box_size // 2
    y1 = cy - box_size // 2
    x2 = cx + box_size // 2
    y2 = cy + box_size // 2

    roi = frame[y1:y2, x1:x2]

    display = frame.copy()
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display, f"Letter: {current_letter} | Captured: {count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Data Collection", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord(' ') and roi.size > 0:
        letter_dir = os.path.join(save_dir, current_letter)
        os.makedirs(letter_dir, exist_ok=True)
        filename = f"webcam_{current_letter}_{count:04d}.jpg"
        cv2.imwrite(os.path.join(letter_dir, filename), roi)
        count += 1
        print(f"Saved {filename}")
    elif chr(key).upper() in "ABCDEFGHIKLMNOPQRSTUVWXY":
        current_letter = chr(key).upper()
        count = 0

cap.release()
cv2.destroyAllWindows()