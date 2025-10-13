import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, Rescaling, BatchNormalization
from keras._tf_keras.keras.utils import image_dataset_from_directory
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def train_model(epochs=200, batch_size=8, learning_rate=0.001, plot_callback=None, stop_callback=None):
    data_dir = 'images/'
    img_height = 256
    img_width = 256
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

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
        Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    class LivePlotCallback(tf.keras.callbacks.Callback):
        def __init__(self, plot_callback=None, stop_callback=None):
            super().__init__()
            self.plot_callback = plot_callback
            self.stop_callback = stop_callback

        def on_epoch_end(self, epoch, logs=None):
            if self.plot_callback:
                self.plot_callback(epoch, logs)
            
            if self.stop_callback and self.stop_callback():
                print("Early stopping requested by user. Saving model...")
                self.model.stop_training = True

    live_plot = LivePlotCallback(plot_callback=plot_callback, stop_callback=stop_callback)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[live_plot, early_stopping, reduce_lr]
    )

    print("\n--- Evaluating on Test Data ---")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    print("\n--- Saving the model ---")
    model.save('image_classifier_model.h5')
    print("Model saved successfully as image_classifier_model.h5")

    return history

if __name__ == "__main__":
    train_model()