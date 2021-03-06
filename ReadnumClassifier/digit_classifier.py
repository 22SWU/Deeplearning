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
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

a = len(train_images)
b = len(test_images)
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

# Evaluate the model using test dataset.
test_loss, test_acc = model.evaluate(test_images, test_labels)  # test_acc를 통해 정확도 나오게
print('Test accuracy:{0} ===> {1}%'.format(test_acc, round(test_acc*100)) ) # round(test_acc*100) = 정확도를 퍼센티지로

# 모델을 사용해 이미지를 어떻게 판별했는지 25개의 이미지 데이터로 보기
# Predict the labels of digit images in our test dataset.
predictions = model.predict(test_images)    # 학습된 모델을 이용해 테스트 데이터 예측

# Then plot the first 25 test images and their predicted labels.
# 결과값 25개만 보여주는 코드
show_sample(
  test_images, 
  ['Predicted: %d' % np.argmax(result) for result in predictions]
  )

# Tensorflow ver.2에서 모델 저장 방식: SavedModel, Keras H5 (SavedModel 권장)
# Save model.
model_path = 'saved_model/mnist_model'
model.save(model_path)
print("DigitClassifier가 SavedModel로 저장되었습니다.")

# 저장된 파일의 정상 작동 확인을 위해 verify_model.py 생성

# tenslorflow lite 모델로 변환하는 코드 추가
# Convert Keras model to TF Lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# mnist.tf 파일에 저장하는 코드
# Save the TF Lite model as file
f = open('mnist.tflite', "wb")
f.write(tflite_model)
f.close()

# 저장된 .tflite 파일이 정상 작동하는지 알아보기 위해 verify_tflite_model.py 생성