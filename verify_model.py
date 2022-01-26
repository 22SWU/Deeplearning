from unittest import TestCase
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import math

print("tensorflow version: ", tf.__version__)
# tensorflow version 확인


# Normalize the input image so that each pixel value is between 0 to 1
# Mnist data set
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 샘플 이미지 보여주는 코드
# Helper function to display digit images
def show_sample(images, labels, sample_count = 25):
    # Create a square with can fit {sample_count} images
    grid_count = math.ceil(math.ceil(math.sqrt(sample_count)))
    grid_count = min(grid_count, len(images), len(labels))
    
    plt.figure(figsize = (2*grid_count, 2*grid_count))
    for i in range(sample_count):
        plt.subplot(grid_count, grid_count, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap = plt.cm.gray)
        plt.xlabel(labels[i])
    plt.show()
    
# 모델이 저장된 위치를 변수에 저장, 저장된 모델 불러오기
model_path = 'saved_model/mnist_model'      # 모델이 저장될 위치 설정
model = tf.keras.models.load_model(model_path)      # 모델 로드
    
# Evaluate the model using test dataset.
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:{0} ===> {1}%'.format(test_acc, round(test_acc*100)))

# Predict the labels of digit images in our test dataset.
predictions = model.predict(test_images)

# Then plot the first 25 test images and their predicted labels.
show_sample(
    test_images,
    ['Predicted: %d' % np.argmax(result) for result in predictions]
)

