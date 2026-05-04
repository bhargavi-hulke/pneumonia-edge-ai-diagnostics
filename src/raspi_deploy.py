import numpy as np
import cv2
import time
import pydicom
import os
import tflite_runtime.interpreter as tflite ## ----when running in raspberry pi it will be this and not simple tensorflow 

MODEL_PATH = "pneumonia_model_quantized.tflite"

TEST_IMAGE = r"C:\work\projects\Signal processing - Pneumonia classification\raw data\jpegs\000db696-cf54-4385-b10b-6b16fbb3f985.dcm"

# loading the model in the raspberry pi
print("Booting TFLite Interpreter...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file {MODEL_PATH} not found!")
    exit()

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f" the signal is from: {os.path.basename(TEST_IMAGE)}")
start_time = time.time()

try:
    if TEST_IMAGE.lower().endswith('.dcm'):
        # 
        ds = pydicom.dcmread(TEST_IMAGE)
        img_raw = ds.pixel_array.astype(float)
    else:
        # 
        img_raw = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            raise FileNotFoundError("OpenCV could not decode the image.")
        img_raw = img_raw.astype(float)
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    exit()

# signal processing on the raspberry end for the input image 
print(" Applying  Filters for processing...")
#  normalization   (values to 0-255 range)
img = (img_raw - np.min(img_raw)) / (np.max(img_raw) - np.min(img_raw)) * 255
img = img.astype(np.uint8)

#  smoothing the raw singal
img = cv2.medianBlur(img, 5)

#  CLAHE 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img)

#  resizing
img = cv2.resize(img, (224, 224))
img_array = np.expand_dims(img, axis=-1)
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)


print("[SYSTEM] Running AI Engine...")
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

probability = prediction[0][0]
end_time = time.time()

# Advisory system 

print("\n" + "="*55)
print(" --- ADVISORY SYSTEM --- ")
print("="*55)
print(f"Inference Latency : {(end_time - start_time)*1000:.1f} ms")
print(f"Signal Magnitude  : {probability:.4f}")
print("-" * 55)

if probability < 0.45:
    print("STATUS: [ HEALTHY ]")
else:
    print("STATUS: [ PNEUMONIA DETECTED- Visit nearest Hospital ]")
    if probability < 0.55:
        print("STAGE : Mild / Early Warning")
    elif probability < 0.85:
        print("STAGE : Moderate")
    else:
        print("STAGE : Severe")
print("="*55 + "\n")
