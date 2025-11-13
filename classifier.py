import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom
from keras._tf_keras.keras.utils import image_dataset_from_directory
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras._tf_keras.keras.applications import VGG16, VGG19, ResNet50
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pathlib

def train_model(epochs=200, batch_size=8, learning_rate=1e-5, plot_callback=None, stop_callback=None):
    data_dir = pathlib.Path('images/')
    img_height = 224
    img_width = 224
    model_save_path = 'image_classifier_model_vgg16.h5'

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)


    all_image_paths = list(data_dir.glob('*/*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]
    class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])
    label_to_index = dict((name, i) for i, name in enumerate(class_names))
    all_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    print(f"Found {len(all_image_paths)} images belonging to {len(class_names)} classes.")
    print("Class names:", class_names)

    from sklearn.model_selection import train_test_split
    
    train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.4, random_state=123, stratify=all_labels)
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_test_paths, val_test_labels, test_size=0.5, random_state=123, stratify=val_test_labels)

    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")
    print(f"Test set size: {len(test_paths)}")
    
    def create_dataset(paths, labels):
        path_ds = tf.data.Dataset.from_tensor_slices(paths)
        
        def load_and_preprocess_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [img_height, img_width])
            return image

        image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        label_ds = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(labels, num_classes=len(class_names)))
        
        return tf.data.Dataset.zip((image_ds, label_ds))

    train_ds = create_dataset(train_paths, train_labels)
    val_ds = create_dataset(val_paths, val_labels)
    test_ds = create_dataset(test_paths, test_labels)

    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    
    train_ds = train_ds.batch(batch_size).map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

    data_augmentation = Sequential(
        [
            RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
            RandomRotation(0.2),
            RandomZoom(0.2),
        ],
        name="data_augmentation"
    )

    base_model = VGG16(input_shape=(img_height, img_width, 3),
                       include_top=False,
                       weights='imagenet')

    base_model.trainable = False

    model = Sequential([
        data_augmentation,
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
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
            message = f"Epoch {epoch+1}: loss={logs['loss']:.4f}, accuracy={logs['accuracy']:.4f}, val_loss={logs['val_loss']:.4f}, val_accuracy={logs['val_accuracy']:.4f}"
            if self.plot_callback:
                self.plot_callback(epoch, logs, message)

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

    print(f"\n--- Saving the model to {model_save_path} ---")
    model.save(model_save_path)
    print(f"Model saved successfully as {model_save_path}")

    return history, model, test_ds, class_names



def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim([min(plt.ylim())-0.05, 1.05])

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.ylim([0, max(plt.ylim())+0.1])
    
    plt.suptitle('Model Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_confusion_matrix(model, test_ds, class_names):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == "__main__":
    training_history, trained_model, test_dataset, class_names = train_model(
        epochs=200,         
        batch_size=8,
        learning_rate=1e-5
    )

    if training_history:
        plot_training_history(training_history)
        plot_confusion_matrix(trained_model, test_dataset, class_names)