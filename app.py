from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from src.train_model import load_and_preprocess_image

app = Flask(__name__)


os.makedirs('static/images', exist_ok=True)


model = None

def load_model_if_exists():
    """Load the model if it exists, otherwise return None"""
    model_path = 'models/fingerprint_model.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

@app.route('/')
def index():
    global model
    if model is None:
        model = load_model_if_exists()
        if model is None:
            return "Please train the model first by running train_model.py", 500
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare_fingerprints():
    global model
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    if 'contact' not in request.files or 'contactless' not in request.files:
        return jsonify({'error': 'Both fingerprint images are required'}), 400
    
    try:
        contact_file = request.files['contact']
        contactless_file = request.files['contactless']
        
        contact_path = os.path.join('static/images', 'temp_contact.png')
        contactless_path = os.path.join('static/images', 'temp_contactless.png')
        
        contact_file.save(contact_path)
        contactless_file.save(contactless_path)
        
 
        contact_img = load_and_preprocess_image(contact_path)
        contactless_img = load_and_preprocess_image(contactless_path)
        

        contact_img = np.expand_dims(contact_img, axis=0)
        contactless_img = np.expand_dims(contactless_img, axis=0)
        

        prediction = float(model.predict([contact_img, contactless_img])[0][0])
        
        os.remove(contact_path)
        os.remove(contactless_path)
        
        return jsonify({
            'probability': prediction,
            'match': bool(prediction > 0.5)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 