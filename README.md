# Skin Cancer Detection System

A deep learning-based system for detecting and classifying skin cancer using the HAM10000 dataset. This project implements a CNN architecture with advanced techniques to achieve high accuracy in skin lesion classification.

## Project Overview

This system uses deep learning to classify skin lesions into different categories of skin cancer, providing a reliable tool for early detection and diagnosis assistance.

### Key Features

- Multi-class classification of skin lesions
- Advanced data augmentation techniques
- Transfer learning with pre-trained models
- Ensemble learning approach
- Real-time prediction capabilities
- Comprehensive model evaluation metrics

## Technologies Used

### Core Technologies
- **Python 3.8+**
- **TensorFlow 2.x**
- **Keras**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Scikit-learn**

### Deep Learning Models
- **EfficientNetB0** (Base model)
- **ResNet50** (Ensemble model)
- **Custom CNN Architecture**

### Data Processing
- **Albumentations** for advanced image augmentation
- **PIL** for image processing
- **Matplotlib** for visualization

## Project Structure

```
skin_cancer_detection/
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model.py           # Model architecture definitions
│   ├── prepare_data.py    # Data preparation and augmentation
│   ├── train.py           # Training pipeline
│   └── utils.py           # Utility functions
├── notebooks/
│   └── skin_cancer_analysis.ipynb  # Analysis and visualization
├── data/                  # Dataset directory (not tracked in git)
├── models/               # Saved model checkpoints
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Methodology

### 1. Data Preprocessing
- Image resizing and normalization
- Advanced data augmentation:
  - Random rotations and flips
  - Color jittering
  - Random brightness/contrast
  - Elastic transformations
  - Cutout augmentation

### 2. Model Architecture
- Base model: EfficientNetB0 with custom head
- Ensemble model: ResNet50
- Custom CNN with:
  - Batch normalization
  - Dropout layers
  - Global average pooling
  - Dense layers with ReLU activation

### 3. Training Strategy
- Transfer learning with fine-tuning
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Cross-validation

### 4. Performance Optimization
- Mixed precision training
- Gradient clipping
- Learning rate warmup
- Model ensembling

## Model Performance

The system achieves:
- Training accuracy: ~95%
- Validation accuracy: ~92%
- Test accuracy: ~90%
- F1-score: ~0.89
- Precision: ~0.91
- Recall: ~0.88

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/kailhashed/Skincancer_detection.git
cd Skincancer_detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the HAM10000 dataset and place it in the `data/` directory

## Usage

1. Prepare the data:
```bash
python src/prepare_data.py
```

2. Train the model:
```bash
python src/train.py
```

3. Use the trained model for predictions:
```python
from src.model import load_model
from src.utils import preprocess_image

model = load_model('models/best_model.h5')
prediction = model.predict(preprocess_image('path_to_image.jpg'))
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HAM10000 dataset creators and contributors
- TensorFlow and Keras teams
- Open source community for various tools and libraries 