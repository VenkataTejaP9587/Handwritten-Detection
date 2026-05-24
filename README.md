# Handwritten Language Detection

A Streamlit-based handwritten language detection system that predicts the language of handwritten text from an uploaded image or camera capture.

## Supported Languages
- English
- Hindi
- Tamil
- Telugu

## Features
- Upload a handwritten image or use camera input
- Predict the language using a trained MobileNetV2-based model
- Display prediction confidence

## Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Files
- `app.py` - Streamlit user interface
- `predict.py` - Image preprocessing and language prediction
- `train_model.py` - Training script for the handwriting language model
- `train_colab.py` - Colab-compatible training helper
- `requirements.txt` - Python dependencies
- `dataset_clean/` - Clean dataset folders for training
- `model/` - Saved model artifacts and class index mapping

## Training

To retrain the model from the cleaned dataset:

```bash
python train_model.py
```

The model and class indices are saved to:

- `model/language_model.h5`
- `model/class_indices.json`

## Notes
- The repository is configured to ignore large data directories and model artifacts.
- If you clone this repo, ensure the `dataset_clean/` and `model/` directories are populated before running the app.

## License
This project does not include a license file by default.
