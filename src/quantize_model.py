import tensorflow as tf

print("Loading heavy AI model...")
model = tf.keras.models.load_model("pneumonia_cnn_model.keras")

print(" Starting TFLite Converter...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# converting the  32-bit floats into 8-bit integers
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print(" Quantizing...")
tflite_model = converter.convert()

print(" Saving lightweight model...")
with open("pneumonia_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print(" SUCCESS! File saved: pneumonia_model_quantized.tflite")
