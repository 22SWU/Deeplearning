import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__.startswith('2')   # tensorflow 버전이 2로 시작한다

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

def showSample(data):       # 화면에 25개의 샘플 이미지를 보여줘라 (25개)
    
    plt.figure(figsize=(10,10))
    for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.numpy(), cmap=plt.cm.gray)
        plt.xlabel(data.index_to_label[label.numpy()])
    plt.show()

image_path = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)

print("======================================")
print("image_path=",image_path)     # 이미지 저장 위치
print("======================================")

# Load input data specific to an on-device ML app.Split it to training data and testing data.
# 학습 이미지 로드
data = DataLoader.from_folder(image_path)   # image_path 에서 데이터 가져와 data 에 할당
train_data, test_data = data.split(0.9) # train_data 90%, 그 중 test_data 는 10%

showSample(data)

print("======================================")
print("The number of Data =", len(data))    # 데이터 총 건수
print("The number of Train data =", len(train_data))    # train_data 건수
print("The number of Test Data =", len(test_data))      # test_data 건수
print("======================================")
