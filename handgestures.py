import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model('hand_gesture_model.keras')
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'unknown']

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Define ROI (center square box)
    x1, y1, x2, y2 = 220, 120, 520, 420  # ROI coordinates
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        continue  # skip this frame if ROI is invalid

    # Preprocess image
    img = cv2.resize(roi, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    label = class_labels[class_index]

    # Display
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"Predicted: {label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
