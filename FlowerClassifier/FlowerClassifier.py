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
    
    
# A helper function that returns 'red'/'black' depending on if its two input
# parameter matches or not.
def get_label_color(val1, val2):
    if val1 == val2:
        return 'black'
    else:
        return 'red'    # 원래 값과 다르게 나왔다

# Then plot 100 test images and their predicted labels.
# If a prediction result is different from the label provided label in "test"
# dataset, we will highlight it in red color.
        
def showPredicted(data):
    plt.figure(figsize=(20, 20))
    predicts = model.predict_top_k(data)
    for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(100)):
        ax = plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image.numpy(), cmap = plt.cm.gray)
        
        predict_label = predicts[i][0][0]
        color = get_label_color(
            predict_label,
            data.index_to_label[label.numpy()]
        )
        ax.xaxis.label.set_color(color)
        plt.xlabel('Predicted: %s' % predict_label)
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

# Customize the TensorFlow model.
# Showed the architecture of model automatically.
# 모델 생성
model = image_classifier.create(train_data)     # epochs는 바꿀 수 있어... (train_data, epochs = 6) 이런 식

# evaluate the model
loss, accuracy = model.evaluate(test_data)
showPredicted(test_data)

# 모델 추출
model.export(export_dir='.', with_metadata=False)    # '.' 의 의미는 현재 데이터 위에 저장된다~