import tensorflow as tf
from tensorflow.keras import layers, models
import os


# This will point to the folder containing folders of processed and sorted data 
DATA_DIR = "C:/work/projects/Signal processing - Pneumonia classification/processed data"

# Standardizing the input size to prevent memory crashes
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32

print("Loading dataset from folders...")


# only 80% of images for the AI to study and learn from
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# the remaining 20%  to test the AI accuracy
val_dataset = tf.keras.utils.image_dataset(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# building the model and cnn network
print("Building the Convolutional Neural Network...")
model = models.Sequential([
    #filter 0: data augmentation, adding just enough noise for ai to learn without overfitting it
   layers.Input(shape=(224, 224, 1)),
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),

    # filter 1: scanning for basic edges and lines like bones
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
   layers.MaxPooling2D((2, 2)),

    # filter 2: scans the  heart curve, lung outlines
     layers.Conv2D(64, (3, 3), activation='relu'),
     layers.MaxPooling2D ((2, 2)),

    # filter 3: scans for complex textures to find cloudy pneumonia fluid
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D ((2, 2)),

    # flatten the 2D data into a 1D decision array
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # drops 50% of connections randomly to prevent overfitting
    layers.Dense(1, activation='sigmoid') # outputs the decimal probability (0.0 to 1.0) 
])

#compiling and training ...
my_optimizer =tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer= my_optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Starting the Training Loop...")
# epochs=10 , that is the model will reviw the dataset 10 python times 
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# saving the model
model.save("pneumonia_cnn_model.keras")
print("Training Complete... model saved as 'pneumonia_cnn_model.keras'")