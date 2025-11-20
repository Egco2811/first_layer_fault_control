import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness, Input, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.regularizers import l2
import numpy as np
from sklearn.metrics import confusion_matrix
import pathlib
import json

def train_model(epochs=200, batch_size=8, learning_rate=1e-5, plot_callback=None, stop_callback=None):
    data_dir = pathlib.Path('images/')
    img_height = 224
    img_width = 224
    model_save_path = 'image_classifier_model_vgg19.keras'
    classes_save_path = 'class_names.json'

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=1e-7)

    all_image_paths = list(data_dir.glob('*/*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]

    class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])
    
    with open(classes_save_path, 'w') as f:
        json.dump(class_names, f)

    label_to_index = dict((name, i) for i, name in enumerate(class_names))
    all_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    print(f"Total images: {len(all_image_paths)}")

    from sklearn.model_selection import train_test_split
    train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.4, random_state=42, stratify=all_labels)
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_test_paths, val_test_labels, test_size=0.5, random_state=42, stratify=val_test_labels)

    def create_dataset(paths, labels):
        path_ds = tf.data.Dataset.from_tensor_slices(paths)
        def load_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [img_height, img_width])
            return image
        image_ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(labels, num_classes=len(class_names)))
        return tf.data.Dataset.zip((image_ds, label_ds))

    train_ds = create_dataset(train_paths, train_labels)
    val_ds = create_dataset(val_paths, val_labels)
    test_ds = create_dataset(test_paths, test_labels)

    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    data_augmentation = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.25),
        RandomZoom(0.25),
        RandomContrast(0.2),
        RandomBrightness(0.2), 
    ], name="augmentation")

    base_model = VGG19(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = Lambda(preprocess_input)(x) 
    x = base_model(x, training=False)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.6)(x)
    outputs = Dense(len(class_names), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    class LivePlotCallback(tf.keras.callbacks.Callback):
        def __init__(self, plot_callback=None, stop_callback=None):
            super().__init__()
            self.plot_callback = plot_callback
            self.stop_callback = stop_callback

        def on_epoch_end(self, epoch, logs=None):
            message = f"Epoch {epoch+1}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}, val_loss={logs['val_loss']:.4f}, val_acc={logs['val_accuracy']:.4f}"
            if self.plot_callback:
                self.plot_callback(epoch, logs, message)
            if self.stop_callback and self.stop_callback():
                self.model.stop_training = True

    print("Starting training VGG19...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[LivePlotCallback(plot_callback, stop_callback), early_stopping, reduce_lr]
    )

    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    return model, test_ds, class_names

def generate_confusion_matrix_data(model, test_ds):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
    
    cm = confusion_matrix(y_true, y_pred)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    return cm, test_acc, test_loss