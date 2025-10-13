import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, Rescaling
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt


data_dir = 'images/'

img_height = 256
img_width = 256
batch_size = 8 

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.3, 
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

val_test_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds = val_test_ds.take(len(val_test_ds) // 2)
test_ds = val_test_ds.skip(len(val_test_ds) // 2)


class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names)
print(f"Found {num_classes} classes.")



data_augmentation = Sequential(
    [
        RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        
    ],
    name="data_augmentation"
)


model = Sequential([
    data_augmentation,

    Rescaling(1./255),

    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(128, activation='relu'),
    
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.summary()



epochs = 200 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


print("\n--- Evaluating on Test Data ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

print("\n--- Saving the model ---")
model.save('image_classifier_model.h5')
print("Model saved successfully as image_classifier_model.h5")