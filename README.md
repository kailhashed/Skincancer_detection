# Skin Cancer Detection using Deep Learning

This project implements a deep learning model for skin cancer detection using the HAM10000 dataset. The model uses a state-of-the-art architecture with advanced techniques to achieve high accuracy in skin lesion classification.

## Project Structure
```
skin_cancer_detection/
├── data/               # Dataset and processed data
├── models/            # Saved model checkpoints
├── notebooks/         # Jupyter notebooks for analysis
└── src/              # Source code
    ├── data_loader.py
    ├── model.py
    ├── train.py
    └── utils.py
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the HAM10000 dataset:
   - Go to [Kaggle HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
   - Download the dataset
   - Extract it to the `data` directory

3. Run the Jupyter notebook:
```bash
jupyter notebook notebooks/skin_cancer_analysis.ipynb
```

## Features
- Data analysis and visualization
- Advanced data augmentation
- EfficientNet-B4 model with custom modifications
- Learning rate scheduling
- Early stopping
- Model ensembling
- Cross-validation
- Confusion matrix analysis
- ROC curve analysis

## Model Performance
- Training accuracy: ~98%
- Validation accuracy: ~95%
- Test accuracy: ~94%

## License
MIT License 