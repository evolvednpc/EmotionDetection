from flask import Flask, render_template, Response, jsonify
import cv2
import time
import json
import numpy as np
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input

app = Flask(__name__)

# Load the emotion detection model and song suggestions
detection_model_path = './trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = './trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

# Load song suggestions from a JSON file
def load_song_suggestions(json_file):
    with open(json_file, 'r') as f:
        song_data = json.load(f)
    return song_data

song_data = load_song_suggestions('mood_songs.json')

# Suggest a song based on the current emotion
def suggest_song(emotion, song_data):
    songs = song_data.get(emotion, [])
    if songs:
        return np.random.choice(songs)
    return None

# Global variables to store the current emotion and song suggestion
current_emotion = ""
current_song_suggestion = ""

# Webcam video capture function
def generate_frames():
    global current_emotion, current_song_suggestion
    video_capture = cv2.VideoCapture(0)
    previous_emotion = None
    emotion_start_time = 0
    last_song_suggestion_time = 0  # Track time of the last song suggestion
    consistent_emotion_duration = 2  # Duration in seconds to confirm emotion
    cool_down_period = 30  # Cooldown period to suggest a new song in seconds

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(face_detection, gray_image)

            detected_emotion = None

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
                detected_emotion = emotion_labels[emotion_label_arg]

                # Draw bounding box and emotion text on the frame
                draw_bounding_box(face_coordinates, frame, (0, 255, 0))
                draw_text(face_coordinates, frame, detected_emotion, (0, 255, 0), 0, -45, 1, 1)

            # Update global emotion and song suggestion if emotion is detected
            if detected_emotion is not None:
                current_time = time.time()

                # Check if detected emotion is consistent
                if detected_emotion == previous_emotion:
                    elapsed_time = current_time - emotion_start_time

                    if elapsed_time >= consistent_emotion_duration:
                        # Check if enough time has passed since the last song suggestion
                        time_since_last_suggestion = current_time - last_song_suggestion_time
                        if time_since_last_suggestion >= cool_down_period:
                            current_song_suggestion = suggest_song(detected_emotion, song_data)
                            if current_song_suggestion:
                                last_song_suggestion_time = current_time  # Update the last suggestion time
                else:
                    # New emotion detected, reset timer
                    previous_emotion = detected_emotion
                    emotion_start_time = current_time

                # Update global emotion
                current_emotion = detected_emotion

            # Encode the frame to display on the web page
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()
    cv2.destroyAllWindows()

# Endpoint to fetch the current emotion and song suggestion
@app.route('/get_emotion_data')
def get_emotion_data():
    return jsonify({
        "emotion": current_emotion,
        "song": current_song_suggestion
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
