import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from utils import extract_keypoints, draw_styled_landmarks, mediapipe_detection

# 4 actions: ok, thank you, hello, goodbye
actions = np.array(['one', 'two', 'three', 'four', 'five', 'fist'])
DATA_PATH = os.path.join('../data')
no_sequences = 30
sequence_length = 30
start_folder = 0

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for sequence in range(start_folder, start_folder + no_sequences):
        try:
            os.makedirs(os.path.join(action_path, str(sequence)))
        except FileExistsError:
            pass


# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up MediaPipe Holistic model
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(start_folder, start_folder + no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    # cv2.waitKey(2000)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()

# Preprocess data and create labels and features
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and train LSTM Neural Network
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(no_sequences, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# Make predictions
res = model.predict(X_test)
predicted_actions = [actions[np.argmax(pred)] for pred in res]
true_actions = [actions[np.argmax(label)] for label in y_test]

# Save weights and model
model.save('../models/main.h5')

# Evaluation using Confusion Matrix and Accuracy
y_true_idx = np.argmax(y_test, axis=1).tolist()
y_pred_idx = np.argmax(res, axis=1).tolist()

conf_matrix = multilabel_confusion_matrix(y_true_idx, y_pred_idx)
accuracy = accuracy_score(y_true_idx, y_pred_idx)

print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
