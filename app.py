from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import base64

app = Flask(__name__)
CORS(app)  # allows your website to call this API

# Load model once at startup
import keras
model = keras.models.load_model('letter_model.keras')
LABELS = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
          'del','nothing','space']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        coords = np.array([[l.x, l.y, l.z] for l in lm]).flatten()
        coords -= coords[:3].mean()
        return coords, result
    return None, None

def draw_landmarks(img, result):
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Decode image from base64
        img_b64 = request.json['image']
        img_data = base64.b64decode(img_b64)
        arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Extract landmarks
        landmarks, result = extract_landmarks(img)

        if landmarks is None:
            return jsonify({
                'letter': 'nothing',
                'confidence': 0,
                'annotated_image': None,
                'hand_detected': False
            })

        # Predict
        pred = model.predict(np.array([landmarks]), verbose=0)
        confidence = float(np.max(pred) * 100)
        letter = LABELS[np.argmax(pred)]

        # Draw landmarks on image
        annotated = draw_landmarks(img.copy(), result)
        _, buffer = cv2.imencode('.jpg', annotated)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'letter': letter,
            'confidence': round(confidence, 1),
            'annotated_image': annotated_b64,
            'hand_detected': True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'letter_model.keras'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)