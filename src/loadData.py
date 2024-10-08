# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#
# from format import get_dataset, grayscale_images, num_class
#
# # Load dataset
# X, X_test, Y, Y_test = get_dataset()
#
# # Define the model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1 if grayscale_images else 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dense(num_class, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(X, Y, epochs=10, validation_data=(X_test, Y_test))
#
# # Evaluate the model
# test_loss, test_acc = model.evaluate(X_test, Y_test)
# print(f'Test Accuracy: {test_acc}')
#
# # Save the model
# model.save('sign_language_model.h5')
# print('Model saved as sign_language_model.h5')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from format import get_dataset, grayscale_images, num_class

# Load dataset
X, X_test, Y, Y_test = get_dataset()

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1 if grayscale_images else 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_class, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Ensure the input data has the correct shape
if grayscale_images:
    X = X.reshape(X.shape[0], 64, 64, 1)
    X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)
else:
    X = X.reshape(X.shape[0], 64, 64, 3)
    X_test = X_test.reshape(X_test.shape[0], 64, 64, 3)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X)

# Learning rate reduction
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                 patience=3,
                                 verbose=1,
                                 factor=0.5,
                                 min_lr=0.00001)

# Train the model
history = model.fit(datagen.flow(X, Y, batch_size=32),
                    epochs=50,
                    validation_data=(X_test, Y_test),
                    callbacks=[lr_reduction])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test Accuracy: {test_acc}')

# Save the model
model.save('test1.h5')
print('Model saved as sign_language_model.h5')
