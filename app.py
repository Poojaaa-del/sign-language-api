from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
import base64
import keras

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model
model = keras.models.load_model('letter_model.keras')

LABELS = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
          'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
          'del','nothing','space']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    model_complexity=0  # lightest model — saves memory
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    try:
        img_b64 = request.json['image']
        img_data = base64.b64decode(img_b64)
        arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Resize to reduce memory usage
        img = cv2.resize(img, (320, 240))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            return jsonify({
                'letter': 'nothing',
                'confidence': 0,
                'annotated_image': None,
                'hand_detected': False
            })

        # Extract landmarks
        lm = result.multi_hand_landmarks[0].landmark
        coords = np.array([[l.x, l.y, l.z] for l in lm]).flatten()
        coords -= coords[:3].mean()

        # Predict
        pred = model.predict(np.array([coords]), verbose=0)
        confidence = float(np.max(pred) * 100)
        letter = LABELS[np.argmax(pred)]

        # Draw landmarks
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Encode annotated image at lower quality to save memory
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'letter': letter,
            'confidence': round(confidence, 1),
            'annotated_image': annotated_b64,
            'hand_detected': True
        })

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'letter_model.keras'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)