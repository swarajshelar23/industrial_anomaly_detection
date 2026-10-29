# Comparative Analysis of Autoencoder Architectures for Unsupervised Industrial Defect Detection

## 1. Project introduction

This project compares three autoencoder-based architectures for unsupervised industrial anomaly detection on four MVTec AD categories: Bottle, Cable, Capsule, and Hazelnut. The project is designed for local Windows + VS Code development using Python and TensorFlow/Keras.

The goal is to learn a normal visual distribution from defect-free training images and then detect anomalies by measuring reconstruction error between an input image and its reconstruction.

## 2. Problem statement

Industrial inspection pipelines often require identifying defective parts without labeled defect examples. Because defective patterns are rare and diverse, unsupervised methods such as autoencoders are useful. A model trained on nominal products can reconstruct normal appearances well, while anomalous features produce larger reconstruction differences.

## 3. Objectives

1. Detect defects using reconstruction-based anomaly scoring.
2. Train models using only defect-free images.
3. Compare AE-CNN, Denoising AE-CNN, and VAE-CNN.
4. Generate anomaly heatmaps and evaluate localization quality.
5. Save reproducible metrics, plots, and trained models.

## 4. Dataset description

The project uses the official MVTec AD dataset. The dataset root is configurable in `config.py`.

Required categories:
- Bottle
- Cable
- Capsule
- Hazelnut

The code automatically detects the available categories and only loads the selected ones. Training uses defect-free images from `train/good`, while test data includes both normal and defective images.

Important dataset facts:
- Images are organized by category, defect type, and split.
- Ground-truth masks are loaded when available.
- No test images are used for training.
- The dataset root path is not hardcoded throughout the code.

## 5. Data preprocessing

The project uses a reusable preprocessing pipeline that:
- loads RGB images,
- resizes to 128x128 by default,
- normalizes pixel intensities to [0, 1],
- uses consistent preprocessing across all models,
- splits normal training data into train and validation sets,
- keeps official test images separate,
- loads binary masks with nearest-neighbor resizing.

The default preprocessing settings are defined in `config.py` and can be updated for 256x256 or other future experiments.

## 6. AE-CNN architecture

The AE-CNN is a convolutional autoencoder trained to reconstruct normal images.

Text architecture diagram:

Input (128x128x3)
  -> Conv + ReLU -> MaxPool
  -> Conv + ReLU -> MaxPool
  -> Conv + ReLU -> MaxPool
  -> Flatten -> Dense latent vector
  -> Dense -> Reshape
  -> Upsample + Conv
  -> Upsample + Conv
  -> Upsample + Conv
  -> Conv2D (3 channels) + Sigmoid
  -> Reconstructed image

This model is appropriate for anomaly detection because it learns compact representations of normal product appearance and produces larger reconstruction errors when a defect is present.

## 7. Denoising AE-CNN architecture

This model uses the same encoder-decoder structure as the AE-CNN but trains on noisy inputs while reconstructing clean images.

Training setup:
- Add Gaussian noise to the model input only.
- Keep the target as the clean image.
- Use MSE as the reconstruction loss.

This helps the model learn robust latent features and is often more resilient to local perturbations or small sensor noise.

## 8. VAE-CNN architecture

The VAE-CNN includes a convolutional encoder with a latent mean vector and latent log-variance vector, followed by a reparameterization step and convolutional decoder.

The mathematically correct VAE objective is:

VAE loss = reconstruction loss + beta × KL divergence

where:
- reconstruction loss encourages the model to faithfully reconstruct input images,
- KL divergence regularizes the latent distribution so it stays close to a standard normal prior.

This regularization makes the latent space more structured and often provides smoother reconstructions. However, the anomaly score is still reconstruction-based, so its sensitivity is limited by the reconstruction objective and the latent bottleneck.

## 9. Training procedure

For each category and each model:
1. Load the normal training images only.
2. Split them into train/validation data.
3. Train on clean normal images (or noisy inputs for denoising AE).
4. Save the best checkpoint.
5. Record loss curves and training time.
6. Keep official test data separate from all training.

Default training settings:
- IMAGE_SIZE = 128
- BATCH_SIZE = 8
- EPOCHS = 30
- LEARNING_RATE = 0.001
- EARLY_STOPPING_PATIENCE = 5
- RANDOM_SEED = 42

A quick-test mode is available to verify the pipeline on a smaller subset and shorter epoch count.

## 10. Anomaly-detection methodology

For each test image:
1. Pass it through the model.
2. Compute the reconstruction.
3. Compute the pixel-wise squared difference.
4. Aggregate this difference into an image-level anomaly score.
5. Compare the score to a threshold selected from validation data.
6. Generate a heatmap to visualize likely defect locations.

For AE-CNN and Denoising AE-CNN, the reconstruction error map is computed as the squared difference between the input image and the reconstructed image.

For VAE-CNN, reconstruction-based scoring is also used, but it should be interpreted as a practical anomaly signal rather than a strict probabilistic guarantee. Reconstruction error is an anomaly signal, not a guarantee that every defect will be detected correctly.

The threshold is chosen from validation normal images using a configurable percentile value and is not tuned on the test set.

## 11. Evaluation metrics

Image-level metrics include:
- ROC-AUC
- precision
- recall
- F1-score
- accuracy
- confusion matrix
- false-positive rate

Pixel-level metrics are computed when masks are available, including:
- pixel ROC-AUC
- pixel precision
- pixel recall
- pixel F1-score
- IoU / Dice-style overlap

The evaluation logic distinguishes between image-level anomaly detection (normal vs anomalous image prediction) and pixel-level anomaly localization (which regions in the image likely contain the defect).

## 12. Model comparison methodology

The project runs the following experiments:
- Bottle: AE-CNN vs Denoising AE-CNN vs VAE-CNN
- Cable: AE-CNN vs Denoising AE-CNN vs VAE-CNN
- Capsule: AE-CNN vs Denoising AE-CNN vs VAE-CNN
- Hazelnut: AE-CNN vs Denoising AE-CNN vs VAE-CNN

This yields 12 category-model evaluation combinations. The table is saved to CSV and used to identify the best-performing model based on actual measured results.

## 13. Installation instructions

### Step 1: Create the project directory

```powershell
mkdir industrial_anomaly_detection
cd industrial_anomaly_detection
```

### Step 2: Open in VS Code

```powershell
code .
```

### Step 3: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 4: Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 14. Execution instructions

### Configure the dataset path

Update `config.py`:

```python
DATASET_ROOT = Path(r"D:\Datasets\mvtec_anomaly_detection")
```

### Validate the dataset

```powershell
python dataset_check.py
```

### Run a quick test

```powershell
python train.py --category bottle --model ae_cnn --quick-test
```

### Train a single model for one category

```powershell
python train.py --category bottle --model ae_cnn
```

### Train all three models for one category

```powershell
python train.py --category bottle --model ae_cnn
python train.py --category bottle --model denoising_ae_cnn
python train.py --category bottle --model vae_cnn
```

### Train all models on all four categories

```powershell
python train.py --category bottle --model ae_cnn
python train.py --category bottle --model denoising_ae_cnn
python train.py --category bottle --model vae_cnn
python train.py --category cable --model ae_cnn
python train.py --category cable --model denoising_ae_cnn
python train.py --category cable --model vae_cnn
python train.py --category capsule --model ae_cnn
python train.py --category capsule --model denoising_ae_cnn
python train.py --category capsule --model vae_cnn
python train.py --category hazelnut --model ae_cnn
python train.py --category hazelnut --model denoising_ae_cnn
python train.py --category hazelnut --model vae_cnn
```

### Evaluate model performance

```powershell
python evaluation.py --category bottle --model ae_cnn
```

### Generate the comparison table

```powershell
python compare_models.py
```

### Launch the optional Streamlit application

```powershell
streamlit run app.py
```

## 15. Output descriptions

The project saves outputs into `outputs/`:

- `dataset_statistics/` — dataset summary CSV
- `models/` — trained `.keras` models
- `histories/` — training history CSV files
- `metrics/` — evaluation metrics CSV files
- `plots/` — training curves, ROC curves, confusion matrices, and model comparisons
- `predictions/` — saved anomaly prediction metadata

## 16. Limitations

- Reconstruction error is a proxy for anomaly likelihood, not a guaranteed defect detector.
- Performance depends strongly on the selected image size, preprocessing choices, and dataset characteristics.
- External images may not match the MVTec distribution and may therefore produce unreliable predictions.
- The project is designed for comparison and academic analysis rather than production inspection deployment.

## 17. Future scope

- Add more model variants and training strategies.
- Evaluate additional categories from MVTec AD.
- Explore anomaly scoring based on latent-space distances or feature-level differences.
- Extend the app with side-by-side model comparisons.
- Add threshold tuning strategies beyond percentile-based selection.

## 18. References

1. Bergmann, P., et al. "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." 2019.
2. Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning.
3. Kingma, D. P., and Welling, M. "Auto-Encoding Variational Bayes." 2013.
4. TensorFlow Keras documentation.
5. MVTec AD dataset documentation: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

## 19. Verified facts vs experimental results

This README separates verified dataset facts from experimental findings. A verified fact is the dataset layout, category names, and available folders. An experimental result is any model performance result produced by training and evaluation in this project. No conclusion about the best model should be stated before the actual experiments are executed.

## 20. Project structure

```text
industrial_anomaly_detection/
├── config.py
├── data_loader.py
├── preprocessing.py
├── models.py
├── train.py
├── anomaly_detection.py
├── evaluation.py
├── visualization.py
├── compare_models.py
├── dataset_check.py
├── app.py
├── requirements.txt
├── README.md
├── outputs/
│   ├── dataset_statistics/
│   ├── models/
│   ├── histories/
│   ├── metrics/
│   ├── plots/
│   │   ├── training_curves/
│   │   ├── roc_curves/
│   │   ├── confusion_matrices/
│   │   ├── model_comparison/
│   │   └── anomaly_heatmaps/
│   └── predictions/
└── .venv/   (optional local environment)
```
