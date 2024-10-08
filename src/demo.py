# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
#
# # Load your pre-trained model
# model = load_model('test1.h5')  # Replace with your model file path
#
# # Define the input size expected by the model
# image_width, image_height = 64, 64  # Replace with your model's input size
#
# def preprocess_frame(frame):
#     frame = cv2.resize(frame, (image_width, image_height))  # Resize to model input size
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     frame = frame / 255.0  # Normalize to [0, 1]
#     frame = frame.reshape((1, image_height, image_width, 1))  # Adjust shape for model input
#     return frame
#
# def main():
#     # Initialize camera
#     cap = cv2.VideoCapture(0)  # 0 for default camera
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
#
#         preprocessed_frame = preprocess_frame(frame)
#         prediction = model.predict(preprocessed_frame)
#         predicted_class = np.argmax(prediction, axis=-1)[0]
#         confidence = np.max(prediction)
#
#         # Display prediction on the frame
#         cv2.putText(frame, f'Predicted Number: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # Show the frame with prediction
#         cv2.imshow('Camera Feed', frame)
#
#         # Exit loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your pre-trained model
model = load_model('test1.h5')  # Replace with your model file path

# Define the input size expected by the model
image_width, image_height = 64, 64  # Replace with your model's input size

def preprocess_frame(frame):
    frame = cv2.resize(frame, (image_width, image_height))  # Resize to model input size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = frame / 255.0  # Normalize to [0, 1]
    frame = frame.reshape((1, image_height, image_width, 1))  # Adjust shape for model input
    return frame

def main():
    cap = cv2.VideoCapture(0)  # 0 for default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame)
        predicted_class = np.argmax(prediction, axis=-1)[0]
        confidence = np.max(prediction)

        # Display prediction on the frame
        cv2.putText(frame, f'Predicted Number: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with prediction
        cv2.imshow('Camera Feed', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
