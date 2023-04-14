import matplotlib
import tensorflow as tf
from matplotlib import pyplot as plt

# matplotlib.use('TkAgg')
from tensorflow.python.keras.models import save_model
from tensorflow.python.keras.saving.save import load_model

preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    zoom_range=0.3,
    vertical_flip=True,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function= preprocess_input
).flow_from_directory(
    directory='train_1',
    target_size=(224, 224),  # resize to this size
    color_mode="rgb",
    class_mode="categorical",
    batch_size=1,
    shuffle=True
)
print(generator.labels)
print(generator.classes)
print(generator.class_indices)

IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)


base_model = tf.keras.applications.MobileNetV3Small(
                                               input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


base_model.trainable = False
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(.2)(x)
predictions = tf.keras.layers.Dense(4, activation='softmax')(x)

# this is the model we will train
model = tf.keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])
h = model.fit(generator, epochs=100,)
model.save("model_cfop.h5")
# model.save("model_hsu.h5")
plt.plot(h.history['accuracy'])
plt.plot(h.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()
model.load_weights("model_cfop.h5")
# model.load_weights("model_hsu.h5")