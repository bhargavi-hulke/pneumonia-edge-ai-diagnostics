import cv2
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
import zipfile  

def process_image(path, target_size=512):
    #  check if the file exists
    if not os.path.exists(path):
        return None, "File Not Found"

    # loading the file with the dataset, see if dcm file or jpg file
    try:
        if path.lower().endswith('.dcm'):
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(float)
        else:
           img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
           if img is None: 
                return None, "Error -  cannot open and read this file format."
           img = img.astype(float)
    except Exception as e:
        return None, f"Load Error: {e}"

    #   normalising the dataset to 0-255 scale for filters
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = img.astype(np.uint8)

    #  auto crop the images / 2d signals 
    try:
        blurred_for_crop = cv2.GaussianBlur(img, (11, 11), 0)
        _, thresh = cv2.threshold(blurred_for_crop, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            # only cropping if it's a big rectangle 
            if cv2.contourArea(c) > (img.shape[0] * img.shape[1] * 0.2):
                x, y, w, h = cv2.boundingRect(c)
                img = img[y:y+h, x:x+w]
    except Exception as e:
        print(f"Crop failed, using full image. Reason: {e}")

    # making the image noise free and moire effect free 
    img = cv2.medianBlur(img, 5) 

    # enhance the image with CLAHE 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # sq. padding
    h, w = img.shape
    side = max(h, w)
    pad_img = np.zeros((side, side), dtype=np.uint8)
    y_off, x_off = (side-h)//2, (side-w)//2
    pad_img[y_off:y_off+h, x_off:x_off+w] = img
    
    # lastly resizing
    final = cv2.resize(pad_img, (target_size, target_size))
    return final / 255.0, "Success"


def run_mass(raw_folder, processed_folder):
    #  ensuring the output folder exists
    os.makedirs(processed_folder, exist_ok=True)
    
    #  look at every file in the raw data folder
    for filename in os.listdir(raw_folder):
    
        if filename.lower().endswith(('.dcm', '.jpg', '.jpeg')):
            raw_path = os.path.join(raw_folder, filename)
            print(f"Processing File: {filename}...")
            
            #run it for every data in the 
            signal, status = process_image(raw_path)
            
            if signal is not None:
                # 
                save_signal = (signal * 255).astype(np.uint8)
                new_filename = filename.replace('.dcm', '.jpg').replace('.jpeg', '.jpg')
                save_path = os.path.join(processed_folder, new_filename)
                cv2.imwrite(save_path, save_signal)
            else:
                print(f"Skipped {filename}: {status}")

#path setup
PROCESSED_DIR = "C:/work/projects/Signal processing - Pneumonia classification/processed data"

RAW_DIR = "C:/work/projects/Signal processing - Pneumonia classification/raw data/jpegs"

print("--- Starting Pipelines ---")

# Execute 
if os.path.exists(RAW_DIR):
    run_mass(RAW_DIR, PROCESSED_DIR)
else:
    print(f"Could not find raw folder at {RAW_DIR}")

# Execute 
if os.path.exists(ZIP_FILE_PATH):
    process_zip(ZIP_FILE_PATH, PROCESSED_DIR, limit= None)
else:
    print(f"Could not find zip file at {ZIP_FILE_PATH}")

print("---Processing Complete! ---")
