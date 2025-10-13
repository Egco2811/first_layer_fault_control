import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array

class Predictor:
    def __init__(self, model_path='image_classifier_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['high', 'ideal', 'low']
        self.img_height = 256
        self.img_width = 256

    def predict(self, image_path):
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        predictions = self.model.predict(img_batch)
        score = tf.nn.softmax(predictions[0])
        predicted_class = self.class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        return predicted_class, confidence


model = tf.keras.models.load_model('image_classifier_model.h5')
print("Model loaded successfully.")

class_names = ['high', 'ideal', 'low'] 


image_path = 'images/low/8.jpg'

img_height = 256
img_width = 256

img = load_img(image_path, target_size=(img_height, img_width))

img_array = img_to_array(img)

img_batch = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_batch)
score = tf.nn.softmax(predictions[0]) 

predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"\nThis image most likely belongs to the '{predicted_class}' class with a {confidence:.2f}% confidence.")