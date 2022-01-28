import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import math

print("tensorflow version: ", tf.__version__)

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
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.xlabel(labels[i])
    plt.show()
    
# 0(zero)으로 보이는 이미지를 다운받에 모델에서 예측하는거 추가
# Download a test image
zero_img_path = keras.utils.get_file(
    'zero.png',
    'https://storage.googleapis.com/khanhlvg-public.appspot.com/digit-classifier/zero.png'
)

image = keras.preprocessing.image.load_img(
    zero_img_path,
    color_mode = 'grayscale',
    target_size = (28,28),
    interpolation = 'bilinear'
)

# Pre-process the image: Adding batch dimension and normalize the pixel value to [0..1]
# In training, we feed images in a batch to the model to imporove training speed, making the model input shape to be (BATCH_SIZE, 28, 28).
# For inference, we still need to match the input shape with training, so we expand the input dimensions to (1, 28, 28) using np.expand_dims
input_image = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, 0)

# Show the pre-processed input image
show_sample(input_image, ['Input Image'], 1)

# TF Lite path 설정
# Read lite model for tensorflow
# tflite_model = None
tflite_path = "mnist.tflite"
with open(tflite_path, 'rb') as f:
    tflite_model = f.read()
# 데이터 읽어서 집어넣기

# tensorflow lite 에서 모델을 실행하기 위해 interpreter 사용
# Run inference with TensorFlow Lite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_image)    # input
interpreter.invoke()    # interpreter 실행 코드
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()[0]      # output

# Print the model's classification result
digit = np.argmax(output)   # 실제 값 digit에 저장

show_sample(input_image, ['Output={}'.format(digit)], 1)
print('Predicted Digit: {0}\nConfidence: {1} ===> {2}%'.format(digit, output[digit], round(output[digit]*100)))