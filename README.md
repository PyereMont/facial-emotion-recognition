# 🎭 Emotion Detection with Deep Learning

Real-time facial emotion recognition system using Convolutional Neural Networks (CNN). This project classifies human expressions into 7 emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Project Overview

This project demonstrates end-to-end deep learning workflow from data exploration to real-time deployment. The CNN model achieves **60% accuracy** on the FER-2013 dataset, with particularly strong performance on happy (82% recall) and surprise (82% recall) emotions.

### Key Features

- 📊 Comprehensive exploratory data analysis
- 🏗️ Custom CNN architecture with 6M parameters
- ⚖️ Class weighting to handle imbalanced dataset
- 🔄 Data augmentation for better generalization
- 📈 Detailed evaluation with confusion matrix
- 🎥 Real-time emotion detection with webcam
- 📝 Well-documented Jupyter notebooks

## 🚀 Demo

![Demo GIF](images/demo.gif)

*Real-time emotion detection using webcam*

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 60.06% |
| Happy Recall | 82% |
| Surprise Recall | 82% |
| Disgust Recall | 72% |

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Data Analysis:** NumPy, Pandas, Matplotlib, Seaborn
- **ML Tools:** scikit-learn

## 📁 Project Structure
```
emotion-detection-cnn/
│
├── notebooks/
│   ├── 01_exploracion_datos.ipynb          # Data exploration
│   ├── 02_entrenamiento_modelo.ipynb       # Model training
│   └── 03_deteccion_tiempo_real.ipynb      # Real-time detection
│
├── src/
│   ├── model.py                             # Model architecture
│   ├── preprocessing.py                     # Data preprocessing
│   └── detector.py                          # Real-time detector
│
├── models/                                  # Trained models (not uploaded)
├── data/                                    # Dataset (not uploaded)
├── images/                                  # Screenshots and results
├── requirements.txt                         # Dependencies
└── README.md                                # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/TU_USUARIO/emotion-detection-cnn.git
cd emotion-detection-cnn
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download FER-2013 dataset:
   - From [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
   - Extract to `data/` folder

## 🎮 Usage

### Training the Model
```bash
# Open and run the training notebook
jupyter notebook notebooks/02_entrenamiento_modelo.ipynb
```

### Real-time Detection
```bash
# Open and run the detection notebook
jupyter notebook notebooks/03_deteccion_tiempo_real.ipynb
```

Or use the standalone script:
```python
from src.detector import EmotionDetector

detector = EmotionDetector(model_path='models/best_model_v2.keras')
detector.run()
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot
- Press `ESC` to exit

## 📈 Model Architecture
```
Input (48x48x1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(256) → BatchNorm → Conv2D(256) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(7, softmax)
```

**Total Parameters:** 5,984,263

## 💡 Key Learnings

1. **Class Imbalance:** Implemented class weights (disgust: 9.4x, happy: 0.57x) to handle severe imbalance
2. **Data Augmentation:** Rotation, zoom, and shifts improved generalization
3. **Regularization:** BatchNormalization and Dropout prevented overfitting
4. **Iterative Improvement:** Model V2 improved from 39% to 60% accuracy

## 🌟 Applications

- **UX Research:** Evaluate emotional reactions during user testing
- **Marketing:** Analyze responses to advertising content
- **Customer Service:** Detect dissatisfaction in real-time
- **Mental Health:** Monitor emotional states during therapy
- **Education:** Identify confusion or disinterest in students
- **HR:** Analyze interviews and performance evaluations

## 🔮 Future Improvements

- [ ] Transfer learning with VGG16/ResNet
- [ ] Ensemble methods for higher accuracy
- [ ] Additional datasets (CK+, AffectNet)
- [ ] Vision Transformers architecture
- [ ] REST API deployment
- [ ] Mobile app with TensorFlow Lite
- [ ] Multi-face tracking improvements

## 📚 Dataset

**FER-2013** (Facial Expression Recognition Challenge)
- 35,887 grayscale images (48x48 pixels)
- 7 emotion categories
- Source: [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**[Tu Nombre]**

- LinkedIn: www.linkedin.com/in/patrick-yeremy-montecinos-alarcon
- Kaggle: https://www.kaggle.com/patrickmontecinos

## 🙏 Acknowledgments

- FER-2013 dataset creators
- TensorFlow and Keras teams
- OpenCV community

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

*Made with ❤️ for the Data Science community*