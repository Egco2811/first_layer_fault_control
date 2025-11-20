import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input 

class Predictor:
    def __init__(self, model_path='image_classifier_model_vgg19.keras', class_file='class_names.json'):
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

        if os.path.exists(self.class_file):
            try:
                with open(self.class_file, 'r') as f:
                    self.class_names = json.load(f)
            except Exception as e:
                print(f"[Predictor] Failed to load class JSON: {e}")
        else:
            print(f"[Predictor] Class file not found: {self.class_file}")

        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(
                    self.model_path, 
                    custom_objects={'preprocess_input': preprocess_input}
                )
                print(f"[Predictor] VGG19 Model loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"[Predictor] Failed to load model: {e}")
                print("[Predictor] PLEASE RETRAIN THE MODEL to fix file corruption.")
        else:
            print(f"[Predictor] Model file not found: {self.model_path}")

    def reload(self):
        print("[Predictor] Reloading resources...")
        self._load_resources()

    def is_ready(self):
        return self.model is not None and len(self.class_names) > 0

    def predict(self, image_path):
        if not self.is_ready():
            self.reload()

        if not self.is_ready():
            return "Error: Model corrupt or missing. Please Train.", 0.0

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
            return "Unknown Class", 0.0
        except Exception as e:
            return f"Prediction Error: {str(e)}", 0.0

if __name__ == "__main__":
    pred = Predictor()