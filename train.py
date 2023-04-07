import matplotlib
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras import layers

# matplotlib.use('TkAgg')

generator = ImageDataGenerator(
    zoom_range=0.1,
    width_shift_range=0.05,
    rescale=1 / 255,
    height_shift_range=0.05,
).flow_from_directory(
    directory='train_2',
    target_size=(64, 40),  # resize to this size
    color_mode="rgb",
    class_mode="categorical",
    batch_size=1,
    shuffle=True
)
print(generator.labels)
print(generator.classes)
print(generator.class_indices)

# for i in range(1, 8):
#
#   plt.subplot(1, 8, i)
#   batch = generator.next()[0]
#   image_ = batch[0].astype('float32')
#   plt.imshow(image_)
# plt.show()

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(64, 40, 3)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),  # applying filter
        layers.MaxPooling2D(pool_size=(2, 2)),  # decreasing size
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),  # applying filter
        layers.MaxPooling2D(pool_size=(2, 2)),  # decreasing size
        layers.Flatten(),
        layers.Dropout(0.5),  # dropping 0
        layers.Dense(3, activation="softmax"),
    ]
)
model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

h = model.fit(generator, epochs=60,)
print(model.to_json())
model.save_weights("model_hsu.h5")
plt.plot(h.history['accuracy'])
plt.plot(h.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()
