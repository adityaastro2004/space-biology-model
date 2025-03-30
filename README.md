# Space Biology Model Zoo

This repository contains two main pipelines: one for text-based space biology information extraction using Transformers (BERT and T5) and another for image captioning using TensorFlow (CNN-LSTM and CNN-Transformer).

---

## Overview

This project aims to develop models that can extract and generate information from space biology data:

- **Text Pipeline:**  
  - Uses **BERT** (e.g., BioBERT) or **T5** (SciFive) to extract details from biomedical texts related to space biology.
  - Includes dataset catalog creation, preprocessing, model training, inference, and evaluation.
  - Experiment tracking is enabled via [Weights & Biases](https://wandb.ai/).

- **Image Pipeline:**  
  - Uses CNN-based architectures combined with LSTM or Transformer layers for image captioning.
  - Includes image data augmentation, text tokenization, model training, evaluation, and caption generation.
  - Models used include EfficientNetB0 and ResNet50V2 for feature extraction.


**Dependencies include:**

- **For Text Pipeline:**
  - `torch`, `transformers`, `datasets`
  - `scikit-learn`, `pandas`, `numpy`
  - `wandb` for experiment tracking

- **For Image Pipeline:**
  - `tensorflow`
  - `matplotlib`, `seaborn`
  - `scikit-learn`, `pandas`, `numpy`
  
---

## Usage

### Text Pipeline (Transformers-based)

1. **Dataset Catalog & Preprocessing:**
   - The code generates CSV catalogs for biomedical and space biology datasets.
   - It simulates pre-training and fine-tuning datasets – feel free to plug in your own data.

2. **Training:**
   - Choose your model type via the `MODEL_TYPE` variable (`bert` or `t5`).

   - The pipeline trains the model, evaluates it, logs results to W&B, and saves a model card along with model artifacts.

3. **Inference:**
   - Use `extract_info_bert` or `extract_info_t5` to extract information from new space biology text samples.

### Image Pipeline (TensorFlow-based)

1. **Data Loading & Preprocessing:**
   - Place your dataset CSV and images in appropriate directories.
   - The script uses `ImageDataGenerator` for data augmentation and tokenizes image captions.

2. **Model Training:**
   - Two models are available:
     - **CNN-LSTM:** Uses EfficientNetB0 for image features combined with LSTM for text decoding.
     - **CNN-Transformer:** Uses ResNet50V2 and a simplified Transformer-based approach.

   - The training function splits validation data for testing and plots training history.

3. **Caption Generation:**
   - Use the `generate_caption` function to generate captions for new images.


---

## Notes and Limitations

- **Data Simulation:**  
  The provided dataset examples are simulated for demonstration. Replace them with your actual space biology datasets for real-world applications.

- **Model Configurations:**  
  The label processing and output parsing in the pipelines are simplified. Adjust the code as necessary for your specific task requirements.

- **Performance:**  
  These models are optimized for space biology text/image data. Performance on other domains may vary—no magic bullet here, just science (and some ML wizardry).

- **Experiment Tracking:**  
  W&B is used to track experiment metrics. Make sure to set up your W&B account if you wish to log experiments.

