from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Import the fashion dataset
# train_images = the 60,000 images used for training
# train_labels = 60,000 labels for classification (0-9 for the type of apparel)
# test_images = the 10,000 images we'll use to test the learning
# test_labels = the classification the AI must give to the images
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# train_images contains the 60,000 training images of 28x28px
print('--train_images--')
print(train_images.shape)
# train_labels contains the 60,000 training labels, each 0-9, corresponding with training images
print('--train_labels--')
print(len(train_labels))
print(train_labels)
# test_images contains the 10,000 testing images
print('--test_images--')
print(test_images.shape)
# test_labels contains the 10,000 testing labels, each 0-9
print('--test_labels--')
print(len(test_labels))
print(test_labels)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
# Pre-process the training set - scale values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0
# Display the first 25 training images and the class beneath them, so we confirm the data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the Neural Network Model
# Flatten - this transforms each images from a 2D[28][28] array to a 1D[784] array
# Dense - builds a neural network layer of 128 nodes which are densely connected to each other
# Dense 2 - builds another layer of 10 nodes, with each node representing a category with a probability as a value
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the Neural Network
# loss - loss function, how accurate the model is during training
# optimizer - optimiser function, which updates the network based on the output of the loss function
# metrics - monitors the training and testing steps, in this case by fraction of images correctly classified
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train it!
# Train the model to associate images with labels, over 5 generations
print('--Training--')
model.fit(train_images, train_labels, epochs=5)
# Test it!
# Test how well the model performs against test set.
# Our accuracy will be less here - this is an example of overfitting, i.e. our model
# has been trained to identify the images in the training set specifically better than other sets
print('--Testing--')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# Make predictions
# Ask our model to predict the label for each image in the test set (i.e. apparel type)
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0])) # 9 - our model thinks the first image is an ankle boot

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
# Finally, let's try and predict a single image
print('--Single Prediction--')
img = test_images[0]
print(img.shape) # Actual image label
# Add the image to a list (since the model is optimised to predict in batch)
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = model.predict(img)
print(predictions_single) # Predicted image label
# Plot the result
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()
# Print the result
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
