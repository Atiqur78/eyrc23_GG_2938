import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import os


# event class names
class_names = ["combat", "destroyedbuilding", "fire", "militaryvehicles", "humanitarianaid"]

# data directories
train_dir = r"D:\Task_2B\training"  # Path to your training data
test_dir = r"D:\Task_2B\testing"  # Path to your test data

# data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# loading and preparing training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# loading and preparing test data

test_data_directory = test_dir
image_files = [os.path.join(test_data_directory, f) for f in os.listdir(test_data_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
images = [img_to_array(load_img(file, target_size=(224, 224))) for file in image_files]
test_data = np.array(images) / 255.0


# defined and compiled the model
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# freezed the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# training the model
history = model.fit(
    train_data,
    epochs=10,
    batch_size=64,
    verbose=1
)

# making predictions
predictions = model.predict(test_data)

# output of event predictions
for i, prediction in enumerate(predictions):
    class_index = tf.argmax(prediction).numpy()
    event = class_names[class_index]
    print(f"Prediction for image {i + 1}: {event}")

# saved the model after training
model.save("event_classification_model.h5")
