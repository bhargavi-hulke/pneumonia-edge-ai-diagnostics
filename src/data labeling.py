import os
import csv
import shutil

def organize_dataset(kaggle_csv, custom_csv, processed_dir):
    # create the AI training folders
    healthy_dir = os.path.join(processed_dir, '0_Healthy')
    pneumonia_dir = os.path.join(processed_dir, '1_Pneumonia')
    
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(pneumonia_dir, exist_ok=True)

    # building a dictionary 
    labels = {}
    
    print("Reading CSV...")
    try:
        with open(kaggle_csv, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                labels[row['patientId']] = int(row['Target'])
    except Exception as e:
        print(f"Failed to read CSV: {e}")

    print("Reading CSV...")
    try:
        with open(custom_csv, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
             
                full_path = row['Path']
                image_name = os.path.basename(full_path) 
                image_id = image_name.replace('.jpg', '').replace('.jpeg', '')
                
                pneumonia_status = row['Pneumonia'].strip()
                if pneumonia_status in ['1', '1.0']:
                    target = 1
                else:
                    target = 0 
                
                labels[image_id] = target
    except Exception as e:
        print(f"Failed to read CSV: {e}")

    # sorting the processed images using the Dictionary
    print("Sorting all images...")
    moved_count = 0
    skipped_count = 0
    
    for filename in os.listdir(processed_dir):
        if not filename.lower().endswith('.jpg'):
            continue
            
        patient_id = filename.replace('.jpg', '')
        
        if patient_id in labels:
            diagnosis = labels[patient_id]
            source_path = os.path.join(processed_dir, filename)
            
            if diagnosis == 1:
                dest_path = os.path.join(pneumonia_dir, filename)
            else:
                dest_path = os.path.join(healthy_dir, filename)
                
            shutil.move(source_path, dest_path)
            moved_count += 1
        else:
            skipped_count += 1


    print(f"\n ....SORTING COMPLETE!")
    print(f"Successfully sorted {moved_count} images into AI training folders.")
    if skipped_count > 0:
        print(f"Left {skipped_count} images in the main folder (Not found in either CSV).")



KAGGLE_CSV = "C:/work/projects/Signal processing - Pneumonia classification/scripts/stage_2_train_labels.csv"
CHex_CSV = "C:/work/projects/Signal processing - Pneumonia classification/scripts/train.csv"

PROCESSED_FOLDER = "C:/work/projects/Signal processing - Pneumonia classification/processed data"

organize_dataset(KAGGLE_CSV, CHex_CSV, PROCESSED_FOLDER)