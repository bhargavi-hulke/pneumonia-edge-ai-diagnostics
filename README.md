# Edge-AI Pneumonia Diagnostic Engine

An optimized edge-AI diagnostic system for pneumonia detection in rural clinics, featuring 16-bit DICOM signal processing and INT8 quantized TFLite inference.An integrated deep learning pipeline for pneumonia detection from chest X-rays, designed specifically to run efficiently on low-power, low-cost hardware like a Raspberry Pi, making it ideal for rural clinics that do not have access to expensive medical servers or stable internet connections.

**Why I built this**: >  I actually built this project to see if I could take a complex deep-learning model and make it really work on a low-power device like a Raspberry Pi. The goal was really simple - to create something that could potentially help in rural clinics where expensive diagnostic hardware isn't readily available. It was a huge learning curve. Especially dealing with messy real-world data and "Moiré" noise. But, it really showed me the power of combining Signal Processing with AI.

## 🛠 System Architecture

The project implements a multi-stage **Digital Signal Processing (DSP)** pipeline prior to inference to ensure high signal integrity:

1.  **Normalization:** Linear Min-Max scaling to standardize 16-bit DICOM and 8-bit JPEG inputs.
2.  **Denoising:** 5x5 Median Filtering to attenuate salt-and-pepper sensor noise.
3.  **Contrast Enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalization) to maximize the Signal-to-Noise Ratio (SNR) of pulmonary opacities.
4.  **Quantized Inference:** Executing an INT8-quantized TFLite model to minimize CPU instruction cycles and memory footprint.

###⚙️How the System Works (The 4-Stage Pipeline)

## 1. Image Loading, Normalization & Pre-processing (src/data_labelling.py)

The Problem: Medical X-rays come in different formats. Some are high-quality hospital files (.dcm), while others are standard images (.jpg).The Solution: This stage reads both formats and rescales all pixel values to a standard range ($0$ to $255$) so the system treats every image equally.Before the AI looks at the X-ray, we clean the image using two main steps:

      Noise Removal: apply a Median Blur filter to remove random dots and camera sensor noise           while keeping the edges of the lungs sharp.
      Contrast Boost: use CLAHE (Adaptive Histogram Equalization) to brighten up hidden details.        This makes faint, cloudy patterns of pneumonia stand out clearly from the background.
      
## 2. AI Training (src/train_model.py)

The Brain: I built and trained a Convolutional Neural Network (CNN) using TensorFlow.
The Learning Process: The AI looked at thousands of labeled X-rays (Healthy vs. Pneumonia). Using the Adam Optimizer, it automatically learned which visual patterns indicate an infection.

## 3. Model Compression & Optimization (src/quantize_model.py)

The Problem: Standard AI models are too large and slow for a small Raspberry Pi CPU.The Solution: We used INT8 Quantization to convert the model's heavy math from 32-bit floating numbers into simple 8-bit integers. This shrank the model file size by 75% (from ~100MB to ~25MB), allowing it to run in milliseconds on edge hardware.
## 4.Final Live Deployment (src/5_raspi_deploy.py)

This is the main application file that ties everything together on the Raspberry Pi. It processes the incoming X-ray, runs the optimized AI model, and prints a clear safety alert.

## 📊 Performance & Constraints
* **Inference Latency:** <400ms on Raspberry Pi 4.
* **Known Constraint:** The system is susceptible to **Optical Aliasing (Moiré patterns)** when capturing images of digital screens. Direct digital transfer (DICOM/JPEG) is required for diagnostic accuracy.An extra filter for eradicating this effect is still incorporated, although,currently, it works for less intense Moiré patterns only.

## 🚀 Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run the inference engine
python src/raspi_deploy.py

```
