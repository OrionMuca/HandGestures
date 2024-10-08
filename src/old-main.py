import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from utils import mediapipe_detection, extract_keypoints, draw_styled_landmarks

mp_holistic = mp.solutions.holistic  # Holistic model
model = load_model('../models/main.h5')  # Replace with the path to your H5 file
cap = cv2.VideoCapture(0)
actions = np.array(['one', 'two', 'three', 'four', 'five', 'fist', 'ok', 'thank you', 'hello', 'goodbye'])
sequence = []
threshold = 0.5
previous_predicted_word = ''


def prob_viz(res, actions, input_frame, threshold):
    predicted_index = np.argmax(res)
    confidence = res[predicted_index]

    text = f'{actions[predicted_index]} {int(confidence * 100)}%'

    if confidence > threshold:
        cv2.putText(input_frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (245,117,16), 2, cv2.LINE_AA)
    else:
        cv2.putText(input_frame, "Low Confidence", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (245,117,16), 2, cv2.LINE_AA)

    return input_frame, actions[predicted_index]



# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_label_index = np.argmax(res)
            predicted_label = actions[predicted_label_index]

            image, predicted_action = prob_viz(res, actions, image, threshold)

            if predicted_action != previous_predicted_word:
                previous_predicted_word = predicted_action
                print(predicted_action)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
