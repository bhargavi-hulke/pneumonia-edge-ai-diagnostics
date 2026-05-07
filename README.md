# Edge-AI Pneumonia Diagnostic Engine
An optimized edge-AI diagnostic system for pneumonia detection in rural clinics, featuring 16-bit DICOM signal processing and INT8 quantized TFLite inference.An integrated deep learning pipeline for pneumonia detection from chest X-rays, designed specifically for deployment on **Raspberry Pi (ARM architecture)**.

**Why I built this**: >  I actually built this project to see if I could take a complex deep-learning model and make it really work on a low-power device like a Raspberry Pi. The goal was really simple - to create something that could potentially help in rural clinics where expensive diagnostic hardware isn't readily available. It was a huge learning curve. Especially dealing with messy real-world data and "Moiré" noise. But, it really showed me the power of combining Signal Processing with AI.

## 🛠 System Architecture
The project implements a multi-stage **Digital Signal Processing (DSP)** pipeline prior to inference to ensure high signal integrity:

1.  **Normalization:** Linear Min-Max scaling to standardize 16-bit DICOM and 8-bit JPEG inputs.
2.  **Denoising:** 5x5 Median Filtering to attenuate salt-and-pepper sensor noise.
3.  **Contrast Enhancement:** CLAHE (Contrast Limited Adaptive Histogram Equalization) to maximize the Signal-to-Noise Ratio (SNR) of pulmonary opacities.
4.  **Quantized Inference:** Executing an INT8-quantized TFLite model to minimize CPU instruction cycles and memory footprint.

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
