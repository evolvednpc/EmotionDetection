import os
import cv2
import time
import json
import numpy as np
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input

def load_song_suggestions(json_file):
    """Load song suggestions from a JSON file."""
    with open(json_file, 'r') as f:
        song_data = json.load(f)
    return song_data

def suggest_song(emotion, song_data):
    """Suggest a song based on the current emotion."""
    songs = song_data.get(emotion, [])
    if songs:
        return np.random.choice(songs)
    return None

def process_webcam():
    # Parameters for loading models
    detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Load models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # Load song suggestions from JSON file
    song_data = load_song_suggestions('mood_songs.json')

    # Variables to track mood persistence and cool-down period
    previous_emotion = None
    emotion_start_time = 0
    last_song_suggestion_time = 0  # Track time of the last song suggestion
    cool_down_period = 240  # 4 minutes in seconds

    # Start capturing video from the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_detection, gray_image)

        current_emotion = None

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, (0, 0))
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            # Predict emotion
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            current_emotion = emotion_labels[emotion_label_arg]

            # Draw bounding box and emotion text on the frame
            draw_bounding_box(face_coordinates, frame, (0, 255, 0))
            draw_text(face_coordinates, frame, current_emotion, (0, 255, 0), 0, -45, 1, 1)

        # Check if emotion is the same for more than 10 seconds
        if current_emotion is not None:
            current_time = time.time()
            if current_emotion == previous_emotion:
                elapsed_time = current_time - emotion_start_time
                if elapsed_time >= 10:
                    # Check if enough time (4 minutes) has passed since the last song suggestion
                    time_since_last_suggestion = current_time - last_song_suggestion_time
                    if time_since_last_suggestion >= cool_down_period:
                        song = suggest_song(current_emotion, song_data)
                        if song:
                            print(f"Suggesting a song for mood '{current_emotion}': {song}")
                            last_song_suggestion_time = current_time  # Update the last song suggestion time
                    emotion_start_time = time.time()  # Reset the time
            else:
                previous_emotion = current_emotion
                emotion_start_time = time.time()

        # Display the frame with emotion detection
        cv2.imshow('Emotion Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam()
