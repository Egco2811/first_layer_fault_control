import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class Predictor:
    def __init__(self, model_path='image_classifier_model_vgg16.h5', class_file='class_names.json'):
        self.model_path = model_path
        self.class_file = class_file
        self.model = None
        self.class_names = []
        self.img_height = 224
        self.img_width = 224
        self._load_resources()

    def _load_resources(self):
        self.model = None
        self.class_names = []

        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except Exception as e:
                print(f"Failed to load model: {e}")
        
        if os.path.exists(self.class_file):
            try:
                with open(self.class_file, 'r') as f:
                    self.class_names = json.load(f)
            except Exception as e:
                print(f"Failed to load classes: {e}")

    def reload(self):
        print("Reloading predictor resources...")
        self._load_resources()

    def predict(self, image_path):
        if not self.model or not self.class_names:
            return "Error: Model or classes not loaded", 0.0

        try:
            img = load_img(image_path, target_size=(self.img_height, self.img_width))
            img_array = img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            
            
            predictions = self.model.predict(img_batch, verbose=0)
            score = tf.nn.softmax(predictions[0])
            
            index = np.argmax(score)
            if index < len(self.class_names):
                predicted_class = self.class_names[index]
                confidence = 100 * np.max(score)
                return predicted_class, confidence
            return "Unknown", 0.0
        except Exception as e:
            return f"Prediction Error: {str(e)}", 0.0

if __name__ == "__main__":
    pred = Predictor()