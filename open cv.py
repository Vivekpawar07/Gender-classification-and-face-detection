import cv2
import numpy as np
import tensorflow as tf

# Function to preprocess a frame
def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype('float32') / 255.0  # Normalize to [0, 1]
    frame = np.expand_dims(frame, axis=0)    # Add batch dimension
    return frame

# Target size for preprocessing
target_size = (64, 64)
try:
    model = tf.keras.models.load_model('gender_classification.h5')
except Exception as e:
    print(f"Error loading model: {e}")

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained gender classification model
# model = tf.keras.models.load_model('gender_classification.h5')

# Check if the model loaded correctly
if model is None:
    print("Error: Failed to load the model.")
    exit()

# Check if the webcam is opened correctly
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        break

    # Convert the frame to grayscale (Haar Cascade works better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Crop face region from the frame
        face_region = frame[y:y+h, x:x+w]

        # Preprocess the face region
        processed_frame = preprocess_frame(face_region, target_size)

        # try:
        #     # # Make prediction with the model
        #     # prediction = model.predict(processed_frame,batch_size=1)
        #     # predicted_class = "Male" if prediction[0][0] < 0.5 else "Female"
        #
        #     # Display the predicted gender
        #     # cv2.putText(frame, f'Predicted: {predicted_class}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        #
        # except Exception as e:
        #     print(f"Error during prediction: {e}")

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
