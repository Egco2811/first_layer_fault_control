import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, Rescaling
from keras._tf_keras.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

def train_model(epochs=200, plot_callback=None):
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

    # Live plot callback
    class LivePlotCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs):
            self.epochs = epochs
            self.acc = []
            self.val_acc = []
            self.loss = []
            self.val_loss = []
            plt.ion()
            self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))

        def on_epoch_end(self, epoch, logs=None):
            self.acc.append(logs.get('accuracy'))
            self.val_acc.append(logs.get('val_accuracy'))
            self.loss.append(logs.get('loss'))
            self.val_loss.append(logs.get('val_loss'))

            self.axs[0].cla()
            self.axs[0].plot(range(epoch+1), self.acc, label='Training Accuracy')
            self.axs[0].plot(range(epoch+1), self.val_acc, label='Validation Accuracy')
            self.axs[0].legend(loc='lower right')
            self.axs[0].set_title('Training and Validation Accuracy')

            self.axs[1].cla()
            self.axs[1].plot(range(epoch+1), self.loss, label='Training Loss')
            self.axs[1].plot(range(epoch+1), self.val_loss, label='Validation Loss')
            self.axs[1].legend(loc='upper right')
            self.axs[1].set_title('Training and Validation Loss')

            plt.pause(0.01)
            if plot_callback:
                plot_callback(self.fig)

        def on_train_end(self, logs=None):
            plt.ioff()
            plt.show()

    live_plot = LivePlotCallback(epochs)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[live_plot]
    )

    print("\n--- Evaluating on Test Data ---")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print("\n--- Saving the model ---")
    model.save('image_classifier_model.h5')
    print("Model saved successfully as image_classifier_model.h5")

    return history

# Only run training if called explicitly
if __name__ == "__main__":
    train_model()