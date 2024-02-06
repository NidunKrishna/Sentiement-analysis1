import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import warnings
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import datetime
from tensorflow import keras

def view_random_images(target_class, num_images=8):
    target_dir = "C:\\Users\\HP\\Desktop\\emotion\\test"
    target_folder = os.path.join(target_dir, target_class)

    random_images = random.sample(os.listdir(target_folder), num_images)

    plt.figure(figsize=(12, 6))
    for i, image in enumerate(random_images):
        plt.subplot(2, 4, i + 1)
        img = mpimg.imread(os.path.join(target_folder, image))
        plt.imshow(img)
        plt.title(f"{target_class} - {i + 1}")
        plt.axis("off")

    plt.show()

view_random_images("happy", 8)
view_random_images("sad", 8)
view_random_images("angry", 8)
view_random_images("surprise", 8)

target_dir = "C:\\Users\\HP\\Desktop\\emotion\\test"
IMG_SIZE = (224, 224)

total_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=target_dir,
    image_size=IMG_SIZE,
    label_mode="categorical",
    batch_size=32
)

total_samples = total_data.cardinality().numpy()
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size

train_data = total_data.take(train_size)
test_data = total_data.skip(train_size)

print("Number of samples in training set:", train_size)
print("Number of samples in testing set:", test_size)
print(total_data.class_names)

checkpoint_path = "Expresion/checkpoint.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    save_freq="epoch",
    verbose=1
)

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomHeight(0.2),
    layers.RandomWidth(0.2),
], name="data_augmentation")

input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB7(include_top=False)
base_model.trainable = False

inputs = layers.Input(shape=input_shape, name="input_layer")
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
outputs = layers.Dense(6, activation="softmax", dtype='float32', name="output_layer")(x)

model_1 = keras.Model(inputs, outputs)

model_1.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)

model_1.summary()

history_model_1_efficientB7 = model_1.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=5,
    batch_size=32,
    validation_data=test_data,
    validation_steps=len(test_data)
)

def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

plot_loss_curves(history_model_1_efficientB7)

base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(include_top=False)
base_model.trainable = False

input_layer = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = base_model(input_layer)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
outputs = tf.keras.layers.Dense(6, activation="softmax", dtype='float32', name="output_layer")(x)
model_2 = tf.keras.Model(input)

model_2.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model_2.summary()

history_model_2 = model_2.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=15,
    batch_size=32,
    validation_data=test_data,
    validation_steps=len(test_data)
)

plot_loss_curves(history_model_2)

