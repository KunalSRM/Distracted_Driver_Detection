import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model/distracted_driver_model.h5')

class_labels = [
    'Safe driving', 'Texting - right', 'Talking on the phone - right',
    'Texting - left', 'Talking on the phone - left', 'Operating the radio',
    'Drinking', 'Reaching behind', 'Hair and makeup', 'Talking to passenger'
]

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use CAP_DSHOW on Windows

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_name = class_labels[class_index]

    cv2.putText(frame, f'Prediction: {class_name}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Distracted Driver Detection - RealTime', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
