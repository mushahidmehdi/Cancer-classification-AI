import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import shutil
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

random.seed(101)
directory = 'Training_Input'

train = 'data/train/'
test = 'data/test/'
validation = 'data/validation/'

os.makedirs(train + 'benign/', exist_ok=True)
os.makedirs(train + 'malignent/', exist_ok=True)


os.makedirs(test + 'benign/', exist_ok=True)
os.makedirs(test + 'malignent/', exist_ok=True)

os.makedirs(validation + 'benign/', exist_ok=True)
os.makedirs(validation + 'malignent/', exist_ok=True)

train_examples = test_examples = validation_examples = 0

for line in open('GroundTruth.csv').readlines()[1:]:
    split_sent = line.split(',')
    img_name = split_sent[0]
    ben_malig = split_sent[1]

    rand_int = random.random()

    if rand_int < 0.8:
        location = train
        train_examples += 1

    elif rand_int < 0.9:
        location = validation
        validation_examples += 1

    else:
        location = test
        test_examples += 1

    if int(float(ben_malig)) == 0:
        shutil.copy(
            'Training_Input/' + img_name + '.jpg',
            location + 'benign/' + img_name + '.jpg'
        )
    elif int(float(ben_malig)) == 1:
            shutil.copy(
                "Training_Input/" + img_name + '.jpg',
                location + 'malignent/' + img_name + '.jpg'
            )

print(f'The Number of Training Set {train_examples}')
print(f'The Number of Validation Set {validation_examples}')
print(f'The Number of Testing Set {test_examples}')

train_examples = 20237
validation_examples = 2641
test_examples = 2453
Image_height = Image_width = 224
batch_size = 32    

model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    layers.Conv2D(64, 3, padding='same'),
    layers.ReLU(),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 4),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.GlobalAvgPool2D(),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

train_img_gen = ImageDataGenerator(
    rescale=1/255.0, rotation_range=15,
    zoom_range=(0.95, 0.95), horizontal_flip=True,
    vertical_flip=True, data_format="channels_last",
    dtype=tf.float32
)

validation_img_gen = ImageDataGenerator(rescale=1/255.0,  dtype=tf.float32)
test_img_gen = ImageDataGenerator(rescale=1/255.0,  dtype=tf.float32)

train_gen = train_img_gen.flow_from_directory(
    'data/train/',
    target_size=(Image_height, Image_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True,
    seed=101
)
validation_gen = validation_img_gen.flow_from_directory(
    'data/validation/',
    target_size=(Image_height, Image_width),
    color_mode='rgb',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=True,
    seed=101
)

test_gen = test_img_gen.flow_from_directory(
    'data/test/',
    target_size=(Image_height, Image_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True,
    seed=101
)

# Metrics to reduce skewness in the dataset

Metrics = [
    keras.metrics.BinaryAccuracy(),
    keras.metrics.Precision(),
    keras.metrics.Recall(),
    keras.metrics.AUC(),
]


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=[keras.losses.BinaryCrossentropy(from_logits=False)],
    metrics=Metrics
)

model.fit(train_gen,
          epochs=5,
          steps_per_epoch=train_examples//batch_size,
          validation_data=validation_gen,
          validation_steps=validation_examples // batch_size)

model.evaluate(validation_gen)
model.evaluate(test_gen)

# Saving Entire architecture along weight and parameters in defaut TF2.0x format
model.save('saved_model/my_model')
