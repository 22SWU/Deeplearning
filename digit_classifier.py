import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import math

print("tensorflow version: ", tf.__version__)
# Helper function to display digit images
def show_sample(images, labels, sample_count=25):
    # Create a square with can fit {sample_count} images
    grid_count = math.ceil(math.ceil(math.sqrt(sample_count)))
    grid_count = min(grid_count, len(images), len(labels))
    
    plt.figure(figsize=(2*grid_count, 2*grid_count))
    for i in range(sample_count):
        plt.subplot(grid_count, grid_count, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap = plt.cm.gray)
        plt.xlabel(labels[i])
    plt.show()

# Normalize the input image so that each pixel value is between 0 to 1.
# Mnist 받아오기
mnist = keras.datasets.mnist
(train_images, train_labels), (test_imags, test_labels) = mnist.load_data()

a = len(train_images)
b = len(test_imags)
print("학습 이미지 총 수 = {}".format(a))
print("테스트 이미지 총 수 = {}".format(b))

# Show the first 25 images in the training dataset.
# 25개의 샘플 이미지 보여주기
show_sample(
    train_images,
    ['Label: %s' % label for label in train_labels]
)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    
    # Optional: You can replace the dense layer above with the convolution layers below to get higher accuracy.
    # keras.layers.Reshape(target_shape=(28, 28, 1)),
    # keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
    # keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
    # keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # keras.layers.Dropout(0.25),
    # keras.layers.Flatten(input_shape=(28, 28)),
    # keras.layers.Dense(128, activation=tf.nn.relu),
    # keras.layers.Dropout(0.5
    keras.layers.Dense(10)
    ])
    
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the digit classification model
# 학습을 실행하는 부분
model.fit(train_images, train_labels, epochs = 5)
# epochs가 5이기 때문에 5번 실행

# Let's take a closer look at our model structure.
# 모델 구조를 보는 부분
model.summary()

